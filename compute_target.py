#!/usr/bin/env python3
"""
Target Vector Computation and Caching for Qwen3-30B-A3B MoE Knowledge Editing
Based on AlphaEdit's optimization approach with caching support
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import pickle
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from .utils import find_fact_lookup_idx, EditRequest

@dataclass
class TargetOptimizationParams:
    v_lr: float = 1e-1                    # Learning rate for optimization
    v_num_grad_steps: int = 50             # Number of gradient steps
    v_weight_decay: float = 1e-3           # Weight decay
    kl_factor: float = 0.0625              # KL divergence factor (Preserved as requested)
    clamp_norm_factor: float = 4.0         # Clamp norm factor
    v_loss_layer: int = -1                 

@dataclass
class TargetVectorResult:
    """Result of target vector computation"""
    target_vector: torch.Tensor           # The computed target vector
    initial_prob: float                   # Initial probability of target token
    final_prob: float                     # Final probability after optimization
    optimization_steps: int               # Number of optimization steps used
    converged: bool                       # Whether optimization converged
    loss_history: List[float]             # Loss history during optimization

class TargetVectorComputer:
    """Compute and cache target vectors for knowledge editing using subject token positions"""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        cache_dir: str = "./target_vector_cache",
        device: str = "cuda",
        fact_token_strategy: str = "subject_last",
        hparams: Optional[TargetOptimizationParams] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.fact_token_strategy = fact_token_strategy

        # Use provided hyperparameters or default
        self.hparams = hparams or TargetOptimizationParams()

        print(f"Target Vector Computer initialized (Subject Token Mode)")
        print(f"Cache directory: {self.cache_dir}")
        print(f"Device: {self.device}")

    def _get_cache_path(self, case_id: int, layer_idx: int) -> Path:
        """Get cache file path using case_id directly"""
        return self.cache_dir / f"target_vector_layer_{layer_idx}_{case_id}.pkl"
    
    def _compute_single_target_vector(
        self,
        request: EditRequest,
        layer_idx: int
    ) -> TargetVectorResult:
        """
        Compute target vector for a single request
        """
        
        # Format prompt with subject - use AlphaEdit standard approach
        full_prompt = request.prompt.format(request.subject)

        # Tokenize
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)

        # Find subject token positions
        subject_end_pos = find_fact_lookup_idx(
            prompt=request.prompt,
            subject=request.subject,
            tok=self.tokenizer,
            fact_token_strategy=self.fact_token_strategy,
            verbose=False
        )

        # Get target token ID
        target_ids = self.tokenizer(request.target_new, return_tensors="pt").input_ids.to(self.device)
        if target_ids.shape[1] == 0:
            raise ValueError(f"Target string '{request.target_new}' could not be tokenized")
        target_id = target_ids[0, 0]
        
        # Get model dimensions
        if hasattr(self.model.config, 'hidden_size'):
            d_model = self.model.config.hidden_size
        elif hasattr(self.model.config, 'n_embd'):
            d_model = self.model.config.n_embd
        else:
            raise ValueError("Could not determine model hidden size")
        
        print(f"Computing target vector for '{request.target_new}' (token_id: {target_id})")
        print(f"Prompt: {full_prompt}")
        print(f"Subject token position: {subject_end_pos}")
        print(f"Optimization: {self.hparams.v_num_grad_steps} steps, lr={self.hparams.v_lr}")

        with torch.no_grad():
            initial_outputs = self.model(**inputs)
            initial_logits = initial_outputs.logits[0, -1, :]  
            initial_probs = F.softmax(initial_logits, dim=-1)
            initial_prob = initial_probs[target_id].item()

        print(f"Initial target probability: {initial_prob:.6f}")

        delta = torch.zeros(d_model, requires_grad=True, device=self.device)
        target_init = None
        intervene_pos = subject_end_pos
        print(f"Intervention will be applied at subject token position: {intervene_pos}")
        target_layer = self.model.model.layers[layer_idx]  # Edit Layer
        optimizer = torch.optim.Adam([delta], lr=self.hparams.v_lr)
        
        # Freeze model parameters
        original_requires_grad = {}
        for name, param in self.model.named_parameters():
            original_requires_grad[name] = param.requires_grad
            param.requires_grad = False
        
        loss_history = []
        final_prob = initial_prob
        converged = False

        try:
            # Optimization loop
            for step in range(self.hparams.v_num_grad_steps):
                optimizer.zero_grad()
                
                # Forward pass with intervention at subject token position
                def intervention_hook(module, input_args, output):
                    _ = module, input_args
                    nonlocal target_init

                    # Handle different output types
                    if isinstance(output, tuple):
                        actual_output = output[0]
                    else:
                        actual_output = output

                    # Record initial value 
                    if target_init is None:
                        target_init = actual_output[0, intervene_pos, :].detach().clone()
                        print(f"Initial target vector (layer {layer_idx} output at subject pos {intervene_pos}) norm: {target_init.norm().item():.4f}")

                    # Apply intervention 
                    modified_output = actual_output.clone()
                    modified_output[0, intervene_pos, :] += delta

                    if isinstance(output, tuple):
                        return (modified_output,) + output[1:]
                    else:
                        return modified_output

                intervention_hook_handle = target_layer.register_forward_hook(intervention_hook)

                try:
                    outputs = self.model(**inputs)

                    target_logits = outputs.logits[0, -1, :]  
                    log_probs = F.log_softmax(target_logits, dim=-1)

                    nll_loss = -log_probs[target_id]

                    # Weight decay
                    if target_init is not None:
                        weight_decay = self.hparams.v_weight_decay * (
                            torch.norm(delta) / torch.norm(target_init) ** 2
                        )
                    else:
                        weight_decay = self.hparams.v_weight_decay * torch.norm(delta) ** 2

                    loss = nll_loss + weight_decay
                    loss_history.append(loss.item())

                    loss.backward()
                    optimizer.step()

                    if target_init is not None:
                        max_norm = self.hparams.clamp_norm_factor * target_init.norm()
                        if delta.norm() > max_norm:
                            with torch.no_grad():
                                delta.data = delta.data * max_norm / delta.norm()

                    with torch.no_grad():
                        final_probs = F.softmax(target_logits, dim=-1)
                        final_prob = final_probs[target_id].item()

                    print(f"Step {step:2d}: loss={loss.item():.4f}, "
                            f"nll_loss={nll_loss.item():.4f}, "
                            f"weight_decay={weight_decay.item():.4f}, "
                            f"target_prob={final_prob:.6f}, "
                            f"delta_norm={delta.norm().item():.4f}")

                    # Early stopping
                    if loss.item() < 0.01:
                        print(f"Early stopping at step {step}")
                        converged = True
                        break

                finally:
                    intervention_hook_handle.remove()
                
        finally:
            # Restore model parameters
            for name, param in self.model.named_parameters():
                param.requires_grad = original_requires_grad.get(name, True)
        
        if target_init is not None:
            target_vector = target_init + delta.detach()
            print(f"Final results:")
            print(f"  Layer {layer_idx} subject token position {intervene_pos} optimization:")
            print(f"  Initial probability: {initial_prob:.6f}")
            print(f"  Final probability: {final_prob:.6f}")
            print(f"  Improvement: {final_prob/initial_prob:.2f}x")
            print(f"  Target vector (layer {layer_idx} output at subject pos) norm: {target_vector.norm().item():.4f}")
            print(f"  Delta norm: {delta.norm().item():.4f}")
            print(f"  Target vector represents the desired output of layer {layer_idx} at subject token position {intervene_pos}")

            return TargetVectorResult(
                target_vector=target_vector,
                initial_prob=initial_prob,
                final_prob=final_prob,
                optimization_steps=len(loss_history),
                converged=converged,
                loss_history=loss_history
            )
        else:
            raise RuntimeError("Could not compute target vector")
    
    def compute_target_vectors(
        self,
        requests: List[EditRequest],
        layer_idx: int,
        force_recompute: bool = False
    ) -> Dict[int, TargetVectorResult]:
        """
        Compute target vectors for multiple requests with caching
        """
        results = {}

        print(f"Computing target vectors for {len(requests)} requests")
        print(f"Layer: {layer_idx}")

        for request in tqdm(requests, desc="Computing target vectors"):
            cache_path = self._get_cache_path(request.case_id, layer_idx)
    
            if cache_path.exists() and not force_recompute:
                try:
                    with open(cache_path, 'rb') as f:
                        result = pickle.load(f)
                    print(f"✓ Loaded cached target vector for case {request.case_id}")
                    print(f"  Cached result: {result.initial_prob:.6f} -> {result.final_prob:.6f}")
                    results[request.case_id] = result
                    continue
                except Exception as e:
                    print(f"✗ Failed to load cache for case {request.case_id}: {e}")

            result = self._compute_single_target_vector(request, layer_idx)

            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
            print(f"✓ Cached target vector for case {request.case_id}")

            results[request.case_id] = result

        return results