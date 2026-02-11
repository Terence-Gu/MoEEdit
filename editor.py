"""
Main knowledge editor for Qwen3-30B-A3B MoE model (Refactored)
"""

import torch
import torch.nn.functional as F
import math
import pickle
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from .utils import EditRequest, find_fact_lookup_idx
from .optim import BlockCoordinateDescent
from .config import MoEEditConfig
from .compute_target import TargetVectorComputer, TargetOptimizationParams, TargetVectorResult as ComputeTargetVectorResult
from .manager import ProjectionMatrixManager, Qwen3MoEStatisticsCollector, GptOssExpertsForwardKeyCollector,compute_projection_matrices_parallel

class MoEKnowledgeEditor:
    """
    MoE Knowledge Editor
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        config: Optional[MoEEditConfig] = None,
        layer_idx: Optional[int] = None,
        layers: Optional[List[int]] = None,
        device: Optional[str] = None,
        lambda_reg: Optional[float] = None,
        projection_threshold: Optional[float] = None,
        stats_dir: Optional[str] = None,
    ):
        if config is None:
            raise ValueError("Configuration must be provided")

        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or config.device
        self.lambda_reg = lambda_reg or config.lambda_reg
        self.projection_threshold = projection_threshold or config.projection_threshold
        
        # Layer selection
        if layers is not None:
            self.layers = layers
        elif config.layers:
            self.layers = config.layers
        else:
            self.layers = [layer_idx if layer_idx is not None else config.layer_idx]
        
        if not all(0 <= l < config.num_layers for l in self.layers):
            raise ValueError(f"Invalid layer indices: {self.layers}")

        self.layer_idx = self.layers[-1]
        self.target_layer_idx = getattr(config, 'target_layer_for_unified_vector', None) or max(self.layers)

        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.d_model = config.d_model
        self.d_hidden = config.d_hidden

        self.stats_dir = Path(stats_dir or config.stats_dir)
        self.stats_dir.mkdir(exist_ok=True)
        
        model_name = next((x for x in [getattr(config, 'model_name', None), getattr(config, 'model_path', None), getattr(model, 'name_or_path', None)] if x), 'unknown')
        self.model_id = re.sub(r'[^A-Za-z0-9._-]+', '_', Path(str(model_name)).name)
        self.dataset_id = re.sub(r'[^A-Za-z0-9._-]+', '_', str(getattr(config, 'dataset_name', 'default')))
        
        self.target_cache_dir = Path(config.target_cache_dir) / self.model_id / self.dataset_id
        self.target_cache_dir.mkdir(parents=True, exist_ok=True)

        self.fact_token_strategy = getattr(config, 'fact_token', 'subject_last')
        print(f"MoE Editor initialized for layers: {self.layers} (Strategy: {self.fact_token_strategy})")

        self.hparams = TargetOptimizationParams(
            v_lr=getattr(config, 'v_lr', 1e-1),
            v_num_grad_steps=getattr(config, 'v_num_grad_steps', 50),
            v_loss_layer=getattr(config, 'v_loss_layer', -1),
            v_weight_decay=getattr(config, 'v_weight_decay', 1e-3),
            kl_factor=getattr(config, 'kl_factor', 0.0625),
            clamp_norm_factor=getattr(config, 'clamp_norm_factor', 4.0)
        )

        self.layer_data = {}
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        for l_idx in self.layers:
            self._init_single_layer(l_idx)

        self._update_legacy_attributes()
        self.use_identity_projection = not any(not d['use_identity_projection'] for d in self.layer_data.values())
        self.target_results = {}

    def _init_single_layer(self, l_idx: int):
        target_layer = self.model.model.layers[l_idx].mlp
        if not (hasattr(target_layer, 'experts') and (hasattr(target_layer, 'gate') or hasattr(target_layer, 'router'))):
            raise ValueError(f"Layer {l_idx} is not a supported MoE layer")

        expert_dim = getattr(getattr(target_layer, 'experts', None), 'expert_dim', self.d_hidden)
        
        # Projection Manager
        proj_mgr = ProjectionMatrixManager(
            model=self.model, tokenizer=self.tokenizer, target_layer=target_layer,
            num_experts=self.num_experts, expert_dim=expert_dim, layer_idx=l_idx,
            device=self.device, cache_dir=str(Path(getattr(self.config, 'projection_cache_dir', './projection_matrix_cache')) / self.model_id),
            nullspace_threshold=getattr(self.config, 'nullspace_threshold', 2e-2),
            fact_token_strategy=self.fact_token_strategy
        )

        # Optimizer
        optimizer = BlockCoordinateDescent(self.num_experts, self.d_model, expert_dim, self.device, self.lambda_reg)

        # Load Matrices
        matrices = {}
        use_identity = True
        
        if not getattr(self.config, 'use_identity', False):
            loaded = {i: m for i in range(self.num_experts) if (m := proj_mgr._load_projection_matrix(i)) is not None}
            if loaded:
                matrices = loaded
                use_identity = False
                print(f"Loaded {len(loaded)} cached matrices for layer {l_idx}")
        
        if use_identity:
            print(f"Using identity matrices for layer {l_idx}")
            eye = torch.eye(expert_dim, device=self.device, dtype=torch.float32)
            matrices = {i: eye.clone() for i in range(self.num_experts)}

        self.layer_data[l_idx] = {
            'target_layer': target_layer, 'projection_manager': proj_mgr, 'bcd_optimizer': optimizer,
            'projection_matrices': matrices, 'original_projection_matrices': {k: v.clone() for k, v in matrices.items()},
            'use_identity_projection': use_identity, 'expert_dim': expert_dim
        }

    @classmethod
    def from_config(cls, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, config: MoEEditConfig):
        return cls(model=model, tokenizer=tokenizer, config=config)

    def _update_legacy_attributes(self):
        """Backward compatibility for single-layer usage"""
        if not self.layers: return
        last = self.layer_data[self.layers[-1]]
        self.target_layer = last['target_layer']
        self.projection_manager = last['projection_manager']
        self.bcd_optimizer = last['bcd_optimizer']
        self.projection_matrices = last['projection_matrices']
        self.original_projection_matrices = last['original_projection_matrices']
        self.use_identity_projection = last['use_identity_projection']
        if self.projection_matrices:
             self.projection_matrix = next(iter(self.projection_matrices.values()))

    def _get_target_computer(self) -> TargetVectorComputer:
        return TargetVectorComputer(
            model=self.model, tokenizer=self.tokenizer, cache_dir=str(self.target_cache_dir),
            device=self.device, fact_token_strategy=self.fact_token_strategy, hparams=self.hparams
        )

    def compute_unified_target_vectors(self, requests: List[EditRequest], force_recompute: bool = False) -> Dict[int, ComputeTargetVectorResult]:
        print(f"Computing unified target vectors using layer {self.target_layer_idx}")
        computer = self._get_target_computer()
        results = computer.compute_target_vectors(
            requests=requests, layer_idx=self.target_layer_idx, force_recompute=force_recompute
        )
        self.target_results.update(results)
        return results

    def compute_projection_matrix(self, num_samples: int = 5000, use_identity: bool = True):
        self.compute_projection_matrices(num_samples, use_identity, unified=True)
        return self.projection_matrix

    def compute_projection_matrices(self, num_samples: int = 5000, use_identity: bool = True, unified: bool = False, force_recompute: bool = False):
        all_matrices = {}
        
        if use_identity:
            for l_idx in self.layers:
                self.layer_data[l_idx]['use_identity_projection'] = True
                self._reset_matrices_to_identity(l_idx)
        else:
            if unified:
                for l_idx in self.layers:
                    mgr = self.layer_data[l_idx]['projection_manager']
                    unified_P = mgr.get_unified_projection_matrix()
                    if unified_P is None:
                        _ = mgr.compute_expert_projection_matrices(
                            list(range(self.num_experts)), num_samples, force_recompute, "wikipedia_style"
                        )
                        unified_P = mgr.get_unified_projection_matrix() or torch.eye(self.d_hidden, device=self.device)
                    self._set_layer_matrices(l_idx, {i: unified_P.clone() for i in range(self.num_experts)}, use_identity=False)
            else:
                print(f"Parallel projection computation for {len(self.layers)} layers")
                compute_projection_matrices_parallel(
                    [self.layer_data[l]['projection_manager'] for l in self.layers],
                    num_samples=num_samples, dataset_name="wikipedia_style", force_recompute=force_recompute
                )
                for l_idx in self.layers:
                    self._set_layer_matrices(l_idx, self.layer_data[l_idx]['projection_manager'].projection_matrices, use_identity=False)

        # Collect results
        for l_idx in self.layers:
            all_matrices[l_idx] = self.layer_data[l_idx]['projection_matrices']
        
        self._update_legacy_attributes()
        return all_matrices

    def _reset_matrices_to_identity(self, layer_idx):
        dim = self.layer_data[layer_idx]['expert_dim']
        eye = torch.eye(dim, device=self.device, dtype=torch.float32)
        self._set_layer_matrices(layer_idx, {i: eye.clone() for i in range(self.num_experts)}, use_identity=True)

    def _set_layer_matrices(self, layer_idx, matrices, use_identity):
        data = self.layer_data[layer_idx]
        data['projection_matrices'] = matrices
        data['original_projection_matrices'] = {k: v.clone() for k, v in matrices.items()}
        data['use_identity_projection'] = use_identity
        if not use_identity:
            print(f"Layer {layer_idx}: Set {len(matrices)} projection matrices")

    def _load_target_vectors_from_cache(self, requests: List[EditRequest]) -> Dict[int, ComputeTargetVectorResult]:
        results = {}
        missing = []
        
        # Determine cache layer
        ul = getattr(self.config, 'target_layer_for_unified_vector', None)
        cache_layer = ul if isinstance(ul, int) else self.layer_idx
        if ul is not None and cache_layer != self.layer_idx:
            print(f"Using unified target vectors from layer {cache_layer}")

        print(f"Checking cache for {len(requests)} requests...")
        for req in requests:
            p = self.target_cache_dir / f"target_vector_layer_{cache_layer}_{req.case_id}.pkl"
            if p.exists():
                try:
                    with open(p, 'rb') as f:
                        res = pickle.load(f)
                        res.target_vector = res.target_vector.to(self.device)
                        results[req.case_id] = res
                        print(f"✅ Loaded cache for case {req.case_id}")
                except Exception:
                    missing.append(req)
            else:
                missing.append(req)

        # 2. Compute Missing
        if missing:
            print(f"Computing {len(missing)} missing vectors...")
            computer = self._get_target_computer()
            try:
                batch_res = computer.compute_target_vectors(missing, cache_layer, getattr(self.hparams, 'v_loss_layer', None))
                results.update(batch_res)
            except Exception as e:
                print(f"Batch failed: {e}. Retrying individually...")
                for req in missing:
                    if req.case_id not in results:
                        try:
                            one_res = computer.compute_target_vectors([req], cache_layer, getattr(self.hparams, 'v_loss_layer', None))
                            results.update(one_res)
                        except Exception:
                            pass # Fail silently or raise based on pref (orig code raised)

        return results

    def collect_edit_statistics(self, layer_idx: int, requests: List[EditRequest], target_vectors: Dict = None, residual_weight: float = 1.0, verify_targets: bool = False) -> Dict[str, Any]:
        layer_data = self.layer_data[layer_idx]
        target_layer = layer_data['target_layer']
        matrices = layer_data['projection_matrices']
        
        # Prepare Data Containers
        batch_size = len(requests)
        stats = {
            'gating': torch.zeros(batch_size, self.num_experts, device=self.device),
            'keys': [], 'values': [], 'residuals': [], 'targets': [],
            'is_new': [], 'subject_positions': [], 'entity_mapping': [], 'prefix_mapping': []
        }

        # Setup Collectors
        collector = Qwen3MoEStatisticsCollector()
        hooks = collector.create_hooks(target_layer)
        wrapper = None
        if hasattr(target_layer, 'experts') and hasattr(target_layer.experts, 'gate_up_proj'):
            wrapper = GptOssExpertsForwardKeyCollector(target_layer.experts)

        # Target Vectors
        if target_vectors is None:
            target_vectors = self.target_results or self._load_target_vectors_from_cache(requests)
        
        print(f"Collecting stats for layer {layer_idx} ({len(requests)} samples)...")

        try:
            with torch.no_grad():
                for t, req in enumerate(requests):
                    collector.clear()
                    
                    # Target Vector
                    z_vec = target_vectors[req.case_id].target_vector.to(self.device) if req.case_id in target_vectors else torch.zeros(self.d_model, device=self.device)
                    inputs = self.tokenizer(req.prompt.format(req.subject) if '{}' in req.prompt else req.prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)
                    end_pos = find_fact_lookup_idx(req.prompt, req.subject, self.tokenizer, self.fact_token_strategy)
                    start_pos = find_fact_lookup_idx(req.prompt, req.subject, self.tokenizer,"subject_first") if self.fact_token_strategy == "subject_last" else end_pos
                    
                    # Update Stats Metadata
                    stats['is_new'].append(True)
                    stats['entity_mapping'].append(t)
                    stats['prefix_mapping'].append(0)
                    stats['subject_positions'].append((start_pos, end_pos))

                    # Capture Setup
                    if wrapper and end_pos is not None:
                        wrapper.clear()
                        wrapper.set_capture_positions([int(end_pos)])
                        wrapper.attach()

                    outputs = self.model(**inputs, output_hidden_states=True)

                    if collector.router_logits:
                        logits = collector.router_logits[-1][0] if collector.router_logits[-1].dim() == 3 else collector.router_logits[-1]
                        probs = F.softmax(logits[end_pos, :].float() + 10, dim=-1)
                        vals, idxs = torch.topk(probs, self.num_experts_per_tok)
                        norm_vals = vals / vals.sum()
                        for k in range(self.num_experts_per_tok):
                            if idxs[k] < self.num_experts: stats['gating'][t, idxs[k]] = norm_vals[k]

                    keys, values = [], []
                    input_vec = collector.moe_inputs[0][0, end_pos, :].to(self.device) if collector.moe_inputs else None
                    
                    for j in range(self.num_experts):
                        if stats['gating'][t, j] > 0 and input_vec is not None:
                            k_val = None
                            # Standard Expert
                            if hasattr(target_layer, 'experts') and isinstance(target_layer.experts, torch.nn.ModuleList):
                                exp = target_layer.experts[j]
                                if hasattr(exp, 'gate_proj'): # SwiGLU
                                    k_val = F.silu(exp.gate_proj(input_vec)) * exp.up_proj(input_vec)
                                elif hasattr(exp, 'w1'): # Mixtral
                                    k_val = F.silu(exp.w1(input_vec)) * exp.w3(input_vec)
                            
                            # Packed Expert
                            if k_val is None and wrapper:
                                k_val = wrapper.get_key_for_expert_and_flat_index(j, int(end_pos))
                                if k_val is not None: k_val = k_val.to(self.device)

                            # Projection
                            if k_val is not None:
                                k_val = k_val.float()
                                v_val = (matrices[j].to(self.device) @ k_val).to(k_val.dtype) if j in matrices else k_val.clone()
                                keys.append(k_val)
                                values.append(v_val)
                            else:
                                keys.append(None); values.append(None)
                        else:
                            keys.append(None); values.append(None)
                    
                    stats['keys'].append(keys)
                    stats['values'].append(values)

                    # 3. Residual & Target
                    curr_hs = outputs.hidden_states[layer_idx+1][0, end_pos, :]
                    stats['residuals'].append((curr_hs - z_vec.float()) * residual_weight)
                    stats['targets'].append(self.tokenizer(req.target_new, return_tensors="pt").input_ids.clone().detach())

        finally:
            for h in hooks: h.remove()
            if wrapper: wrapper.detach()

        return stats

    def optimize(self, layer_idx: int, stats: Dict, method: str = 'bcd', **kwargs):
        return self.layer_data[layer_idx]['bcd_optimizer'].optimize(
            stats, num_passes=kwargs.get('num_passes', 2), 
            projection_matrices=self.layer_data[layer_idx]['projection_matrices']
        )

    def apply_weight_updates(self, layer_idx: int, deltas: List[torch.Tensor], use_expert_projection: bool = True):
        print(f"Applying updates to layer {layer_idx}")
        ld = self.layer_data[layer_idx]
        target_layer, matrices = ld['target_layer'], ld['projection_matrices']
        use_id = ld['use_identity_projection']
        
        with torch.no_grad():
            cnt = 0
            experts_mod = getattr(target_layer, 'experts', None)
            
            # Packed Experts
            if hasattr(experts_mod, 'down_proj') and isinstance(experts_mod.down_proj, torch.nn.Parameter):
                dp = experts_mod.down_proj
                for j, delta in enumerate(deltas):
                    if delta.norm() < 1e-6: continue
                    upd = (delta @ matrices[j].to(dp.device) if use_expert_projection and j in matrices and not use_id else delta).to(dp.dtype)
                    dp.data[j].add_(upd.T.contiguous())
                    cnt += 1
            
            # ModuleList Experts
            elif hasattr(experts_mod, '__len__'):
                for j, delta in enumerate(deltas):
                    if delta.norm() < 1e-6: continue
                    exp = experts_mod[j]
                    if hasattr(exp, 'down_proj') and hasattr(exp.down_proj, 'weight'):
                        w = exp.down_proj.weight
                        upd = (delta @ matrices[j].to(w.device) if use_expert_projection and j in matrices and not use_id else delta).to(w.dtype)
                        w.data.add_(upd)
                        cnt += 1
            
            print(f"Updated {cnt}/{self.num_experts} experts")

    def edit_knowledge(self, requests: List[EditRequest], method: str = None, num_passes: int = None, verify_targets: bool = None):
        # Defaults
        method = method or getattr(self.config, 'method', 'bcd')
        num_passes = num_passes or getattr(self.config, 'num_passes', 2)
        verify_targets = verify_targets or getattr(self.config, 'verify_targets', False)

        print(f"Processing {len(requests)} requests across layers {self.layers}")

        # 1. Target Vectors
        missing = [r for r in requests if r.case_id not in self.target_results]
        if missing:
            print(f"Computing {len(missing)} missing vectors...")
            self.target_results.update(self.compute_unified_target_vectors(missing))

        # 2. Projections
        print("Computing projection matrices...")
        self.compute_projection_matrices(
            getattr(self.config, 'num_samples', 5000), 
            getattr(self.config, 'use_identity', True), 
            getattr(self.config, 'unified', False)
        )

        # 3. Edit Layers
        res = {}
        for i, l_idx in enumerate(self.layers):
            w = 1.0 / math.sqrt(len(self.layers) - i)
            print(f"\nProcessing Layer {l_idx} (w={w:.4f})")
            
            deltas = self._edit_single_layer(l_idx, requests, self.target_results, w, method, num_passes, verify_targets)
            res[l_idx] = deltas
        
        return res

    def _edit_single_layer(self, layer_idx, requests, target_vectors, residual_weight, method, num_passes, verify_targets):
        stats = self.collect_edit_statistics(layer_idx, requests, target_vectors, residual_weight, verify_targets)
        deltas = self.optimize(layer_idx, stats, method, num_passes=num_passes)
        self.apply_weight_updates(layer_idx, deltas, use_expert_projection=not self.layer_data[layer_idx]['use_identity_projection'])
        return deltas