"""
Unified manager for hooks and projection matrices in MoE knowledge editing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
from dataclasses import dataclass
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset as hf_load_dataset


"""
Statistics collection and hook management for MoE layers
"""

class Qwen3MoEStatisticsCollector:
    """Statistics collector for Qwen3 MoE layers"""

    def __init__(self):
        self.clear()

    def clear(self):
        """Clear all collected statistics"""
        self.router_logits = []          # could be full logits or top-k scores depending on model
        self.expert_indices = []         # when router returns (scores, indices), we store indices here
        self.moe_inputs = []     # inputs to the whole MoE layer (used for inferring [B, L] to reshape flattened outputs)

    def create_hooks(self, moe_layer):
        """Create forward hooks for Qwen3/gpt-oss MoE layer (supports gate or router)"""
        hooks = []

        def moe_input_hook(module, input, output):
            # Capture the hidden states entering the MoE layer
            self.moe_inputs.append(input[0].detach())

        hooks.append(moe_layer.register_forward_hook(moe_input_hook))

        # Hook for router (gate) to capture routing decisions
        def gate_hook(module, input, output):
            # Support both tensor and tuple outputs (e.g., gpt-oss returns (scores, indices))
            try:
                if isinstance(output, tuple) and len(output) > 0:
                    logits = output[0]
                    self.router_logits.append(logits.detach() if hasattr(logits, 'detach') else logits)
                    if len(output) > 1:
                        indices = output[1]
                        self.expert_indices.append(indices.detach() if hasattr(indices, 'detach') else indices)
                else:
                    logits = output
                    self.router_logits.append(logits.detach() if hasattr(logits, 'detach') else logits)
            except Exception:
                # Skip if unexpected structure
                pass

        if hasattr(moe_layer, 'gate'):
            hooks.append(moe_layer.gate.register_forward_hook(gate_hook))
        elif hasattr(moe_layer, 'router'):
            hooks.append(moe_layer.router.register_forward_hook(gate_hook))

        return hooks


class GptOssExpertsForwardKeyCollector:
    """Forward wrapper for GptOssExperts to capture gated outputs (keys) before down_proj.
    """
    def __init__(self, experts_module):
        self.experts = experts_module
        self._orig_forward = None
        # Capture: expert_idx -> dict(flat_idx -> tensor[d_k])
        self.captured: Dict[int, Dict[int, torch.Tensor]] = {}
        self.capture_flat_indices: Optional[set[int]] = None
        self.last_batch_size: Optional[int] = None
        self.last_seq_len: Optional[int] = None

    def set_capture_positions(self, flat_indices: List[int]):
        self.capture_flat_indices = set(int(i) for i in flat_indices)

    def clear(self):
        self.captured.clear()
        self.capture_flat_indices = None

    def get_key_for_expert_and_flat_index(self, expert_idx: int, flat_idx: int) -> Optional[torch.Tensor]:
        if expert_idx in self.captured and flat_idx in self.captured[expert_idx]:
            return self.captured[expert_idx][flat_idx]
        return None

    def get_all_keys(self) -> Dict[int, Dict[int, torch.Tensor]]:
        return self.captured

    def attach(self):
        if self._orig_forward is not None:
            return
        self._orig_forward = self.experts.forward

        def wrapped_forward(hidden_states: torch.Tensor, router_indices=None, routing_weights=None) -> torch.Tensor:
            batch_size = hidden_states.shape[0]
            self.last_batch_size = batch_size
            # Flatten tokens to match expert code convention
            hidden_states_flat = hidden_states.reshape(-1, self.experts.hidden_size)
            num_tokens = hidden_states_flat.shape[0]
            num_experts = routing_weights.shape[1] if routing_weights is not None else getattr(self.experts, 'num_experts', 0)
            need_capture = self.capture_flat_indices is not None and len(self.capture_flat_indices) > 0

            
            N = hidden_states_flat.shape[0]
            
            hs_rep = hidden_states_flat.repeat(num_experts, 1).view(num_experts, -1, self.experts.hidden_size)
            
            gate_up = torch.bmm(hs_rep, self.experts.gate_up_proj) + self.experts.gate_up_proj_bias[..., None, :]
            gate, up = gate_up[..., ::2], gate_up[..., 1::2]
            gate = gate.clamp(min=None, max=self.experts.limit)
            up = up.clamp(min=-self.experts.limit, max=self.experts.limit)
            glu = gate * torch.sigmoid(gate * self.experts.alpha)
            gated_output = (up + 1) * glu 

            if need_capture:
                for fidx in list(self.capture_flat_indices):
                    if 0 <= fidx < N:
                        vecs = gated_output[:, fidx, :]  # [num_experts, d_k]
                        for e in range(num_experts):
                            self.captured.setdefault(int(e), {})[int(fidx)] = vecs[e].detach().clone().to(torch.float32).cpu()

            next_states = torch.bmm(gated_output, self.experts.down_proj)
            next_states = next_states + self.experts.down_proj_bias[..., None, :]
            next_states = next_states.view(num_experts, batch_size, -1, self.experts.hidden_size)
            # routing_weights: [N, E] -> transpose -> [E, N] -> view -> [E, B, L]
            next_states = next_states * routing_weights.transpose(0, 1).view(num_experts, batch_size, -1)[..., None]
            
            next_states = next_states.sum(dim=0)
            return next_states

        self.experts.forward = wrapped_forward

    def detach(self):
        if self._orig_forward is not None:
            self.experts.forward = self._orig_forward
            self._orig_forward = None
        # Clear captured state
        self.clear()
class TokenizedDataset(Dataset):
    """
    Converts a dataset of text samples (list of strings or HF dataset) into token sequences.
    Performs truncation to model's max length.
    """
    def __init__(self, text_dataset, tokenizer, maxlen, field="text"):
        self.text_dataset = text_dataset
        self.field = field
        self.tokenizer = tokenizer
        self.maxlen = maxlen

    def __len__(self):
        return len(self.text_dataset)

    def __getitem__(self, i):
        item = self.text_dataset[i]
        if self.field is not None and isinstance(item, dict):
            text = item[self.field]
        else:
            text = item

        token_list = self.tokenizer.encode(
            text, truncation=True, max_length=self.maxlen
        )
        attention_mask = [1] * len(token_list)
        return dict(
            input_ids=torch.tensor(token_list, dtype=torch.long),
            attention_mask=torch.tensor(attention_mask, dtype=torch.long),
        )

def length_collation(token_size):
    def collate_fn(items):
        items = sorted(items, key=lambda x: -len(x["input_ids"]))
        batches = []
        batch = []
        batch_width = 0
        for item in items:
            item_width = len(item["input_ids"])
            if item_width == 0:
                break
            if batch_width * (len(batch) + 1) > token_size:
                batches.append(make_padded_batch(batch))
                batch = []
                batch_width = item_width
            if not batch:
                batch_width = item_width
            batch.append(item)
        if len(batch):
            batches.append(make_padded_batch(batch))
        return batches
    return collate_fn

def make_padded_batch(items):
    max_len = max(len(d["input_ids"]) for d in items)
    if max_len == 0:
        return {k: torch.zeros((0, 0), dtype=torch.long) for k in items[0]}
    
    batch = {}
    for k in items[0].keys():
        batch[k] = pad_sequence([d[k] for d in items], batch_first=True, padding_value=0)
    return batch

class OnlineCovariance:
    """
    Tracks covariance (second moment) online without storing all keys.
    Computes X^T X accumulation.
    Supports CPU offloading to prevent VRAM OOM when processing multiple MoE layers.
    """
    def __init__(self, dim: int, device: str, store_on_cpu: bool = True):
        self.dim = dim
        self.compute_device = device
        self.store_device = 'cpu' if store_on_cpu else device
        self.mom2 = torch.zeros(dim, dim, device=self.store_device, dtype=torch.float32)
        self.count = 0

    def update(self, x: torch.Tensor):
        if x.numel() == 0:
            return
        
        x = x.to(self.compute_device).float()
        batch_cov = x.T @ x
        self.mom2 += batch_cov.to(self.store_device)
        self.count += x.shape[0]

    def get_covariance(self) -> torch.Tensor:
        if self.count == 0:
            return torch.eye(self.dim, device=self.compute_device)
        cov = self.mom2.to(self.compute_device) / self.count
        return cov


class StatsCollector:
    """
    Manages hooks and updates OnlineCovariance trackers.
    """
    def __init__(self, target_layer, expert_indices: List[int], expert_dim: int, device: str, store_on_cpu: bool = True):
        self.target_layer = target_layer
        self.expert_indices = set(expert_indices)
        self.device = device
        self.trackers: Dict[int, OnlineCovariance] = {
            idx: OnlineCovariance(expert_dim, device, store_on_cpu=store_on_cpu) 
            for idx in expert_indices
        }
        self.hooks = []
        self.gpt_oss_wrapper = None
        self.current_mask = None 

    def set_current_batch_mask(self, mask: torch.Tensor):
        self.current_mask = mask.to(self.device)

    def register_hooks(self):
        if self._is_packed_experts():
            print("Detected packed experts (GPT-OSS). Using Forward Wrapper.")
            self._register_gpt_oss_wrapper()
            return

        for expert_idx in self.expert_indices:
            if expert_idx < len(self.target_layer.experts):
                expert = self.target_layer.experts[expert_idx]
                if hasattr(expert, 'down_proj'):
                    self.hooks.append(
                        expert.down_proj.register_forward_hook(self._make_hook(expert_idx))
                    )

    def _make_hook(self, expert_idx):
        def hook(module, input, output):
            data = input[0].detach()
            flat_data = data.view(-1, data.shape[-1])
            self.trackers[expert_idx].update(flat_data)
        return hook

    def _is_packed_experts(self):
        return not hasattr(self.target_layer, 'experts') or \
               (isinstance(self.target_layer.experts, nn.Module) and not isinstance(self.target_layer.experts, nn.ModuleList))

    def _register_gpt_oss_wrapper(self):
        self.gpt_oss_wrapper = GptOssExpertsOnlineCollector(
            self.target_layer.experts, 
            self.trackers,
            self.expert_indices
        )
        self.gpt_oss_wrapper.attach()

    def cleanup(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        if self.gpt_oss_wrapper:
            self.gpt_oss_wrapper.detach()


class GptOssExpertsOnlineCollector:
    """Forward wrapper for GPT-OSS Packed Experts."""
    def __init__(self, experts_module, trackers: Dict[int, OnlineCovariance], target_indices: set):
        self.experts = experts_module
        self.trackers = trackers
        self.target_indices = target_indices
        self._orig_forward = None

    def attach(self):
        if self._orig_forward is not None: return
        self._orig_forward = self.experts.forward
        self.experts.forward = self.wrapped_forward

    def detach(self):
        if self._orig_forward is not None:
            self.experts.forward = self._orig_forward
            self._orig_forward = None

    def wrapped_forward(self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None, **kwargs) -> torch.Tensor:
        try:
            with torch.no_grad():
                hidden_states_flat = hidden_states.view(-1, self.experts.hidden_size)
                
                for expert_idx in self.target_indices:
                    mask = None
                    if router_indices is not None:
                        expert_id_tensor = torch.tensor(expert_idx, device=hidden_states.device)
                        mask = (router_indices == expert_id_tensor).any(dim=-1)
                    
                    if mask is None or mask.sum() == 0:
                        continue
                        
                    tokens = hidden_states_flat[mask]
                    
                    if hasattr(self.experts, 'gate_up_proj'):
                        weights = self.experts.gate_up_proj[expert_idx] 
                        bias = self.experts.gate_up_proj_bias[expert_idx] if hasattr(self.experts, 'gate_up_proj_bias') else None
                        
                        if weights.shape[0] == tokens.shape[-1]:
                            gate_up = torch.matmul(tokens, weights)
                            if bias is not None:
                                gate_up += bias
                        else:
                            gate_up = torch.nn.functional.linear(tokens, weights, bias)
                        
                        gate, up = gate_up.chunk(2, dim=-1)
                        gated_output = torch.nn.functional.silu(gate) * up
                        self.trackers[expert_idx].update(gated_output)
                        
        except Exception:
            pass
            
        return self._orig_forward(hidden_states, router_indices=router_indices, routing_weights=routing_weights, **kwargs)

@dataclass
class ProjectionMatrixCache:
    matrix: torch.Tensor
    expert_idx: int
    computation_time: str
    matrix_shape: Tuple[int, int]

class ProjectionMatrixManager:
    """
    Unified projection matrix manager.
    """
    def __init__(self,
                 model,
                 tokenizer,
                 target_layer,
                 num_experts: int,
                 expert_dim: int,
                 layer_idx: int,
                 device: str = "cuda",
                 cache_dir: str = "./projection_matrix_cache",
                 nullspace_threshold: float = 2e-2,
                 fact_token_strategy: str = "subject_last", 
                 **kwargs):

        self.model = model
        self.tokenizer = tokenizer
        self.target_layer = target_layer
        self.num_experts = num_experts
        self.expert_dim = expert_dim
        self.layer_idx = layer_idx
        self.device = device
        self.nullspace_threshold = nullspace_threshold
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.projection_matrices: Dict[int, torch.Tensor] = {}

    def get_dataset(self, dataset_name: str, num_samples: int) -> Dataset:
        maxlen = 1024
        
        
        local_wiki_path = "path/to/wiki"
        raw_data = None
        field = "text"

        if Path(local_wiki_path).exists():
            try:
                print(f"Found local dataset at {local_wiki_path}")
                raw_data = self._load_from_local_cache(local_wiki_path, num_samples)
                field = None 
            except Exception as e:
                print(f"Failed to load from local path: {e}")
        
        if raw_data is None:
            print("Loading from HuggingFace...")
            ds = hf_load_dataset("wikipedia", "20231101.en", split="train")
            raw_data = ds.select(range(min(len(ds), num_samples)))
            field = "text"
        
        return TokenizedDataset(raw_data, self.tokenizer, maxlen=maxlen, field=field)

    def _load_from_local_cache(self, cache_path: str, num_samples: int) -> List[str]:
        parquet_pattern = f"{cache_path}/**/train-*.parquet"
        parquet_files = glob.glob(parquet_pattern, recursive=True)
        if not parquet_files: raise FileNotFoundError(f"No parquet files found in {cache_path}")
        texts = []
        for parquet_file in sorted(parquet_files):
            if len(texts) >= num_samples: break
            try:
                dataset = hf_load_dataset("parquet", data_files=parquet_file, split="train")
                for sample in dataset:
                    if len(texts) >= num_samples: break
                    text = sample.get("text", "")
                    if len(text.strip()) > 50:
                        texts.append(text.strip()) 
            except Exception: continue
        return texts

    def _get_experts_to_compute(self, expert_indices: List[int], force_recompute: bool) -> List[int]:
        to_compute = []
        for idx in expert_indices:
            if not force_recompute:
                cached = self._load_projection_matrix(idx)
                if cached is not None:
                    self.projection_matrices[idx] = cached
                    continue
            to_compute.append(idx)
        return to_compute

    def _prepare_collector(self, expert_indices: List[int], store_on_cpu: bool = True) -> StatsCollector:
        collector = StatsCollector(
            self.target_layer, 
            expert_indices, 
            self.expert_dim, 
            self.device,
            store_on_cpu=store_on_cpu
        )
        collector.register_hooks()
        return collector

    def _get_cache_path(self, expert_idx: int) -> Path:
        return self.cache_dir / f"proj_mat_layer_{self.layer_idx}_expert_{expert_idx}.pkl"
    
    def _save_projection_matrix(self, expert_idx: int, matrix: torch.Tensor):
        data = ProjectionMatrixCache(matrix.cpu(), expert_idx, datetime.now().isoformat(), matrix.shape)
        with open(self._get_cache_path(expert_idx), 'wb') as f:
            pickle.dump(data, f)

    def _load_projection_matrix(self, expert_idx: int) -> Optional[torch.Tensor]:
        try:
            p = self._get_cache_path(expert_idx)
            if not p.exists(): return None
            with open(p, 'rb') as f:
                data = pickle.load(f)
            return data.matrix.to(self.device)
        except Exception:
            return None
    def _finalize_computation(self, collector: StatsCollector, expert_indices: List[int]) -> Dict[int, torch.Tensor]:
        unified_mom2 = torch.zeros(self.expert_dim, self.expert_dim, device=self.device)
        unified_count = 0

        for expert_idx in expert_indices:
            tracker = collector.trackers[expert_idx]
            
            cov = tracker.get_covariance() 
            unified_mom2 += (cov * tracker.count)
            unified_count += tracker.count
            
            proj = self._compute_projection_from_cov(cov)
            
            self.projection_matrices[expert_idx] = proj
            self._save_projection_matrix(expert_idx, proj)

        if unified_count > 0:
            unified_cov = unified_mom2 / unified_count
            unified_proj = self._compute_projection_from_cov(unified_cov)
            self._save_unified_projection_matrix(unified_proj)
            
        return {idx: self.projection_matrices[idx] for idx in expert_indices}

    def _compute_projection_from_cov(self, cov_matrix: torch.Tensor) -> torch.Tensor:
        try:
            S, U = torch.linalg.eigh(cov_matrix)
            max_eig = S[-1].item()
            if max_eig == 0: 
                return torch.eye(cov_matrix.shape[0], device=self.device)
            
            is_significant = S > self.nullspace_threshold
            U_significant = U[:, is_significant]
            
            P = torch.eye(cov_matrix.shape[0], device=self.device) - U_significant @ U_significant.T 
            # This (P = I - U_{significant}U_{significant}^T) is equivalent to P = U_{null}U_{null}^T
            return P
        except Exception as e:
            print(f"SVD failed: {e}")
            return torch.eye(cov_matrix.shape[0], device=self.device)
        

    def _get_cache_path(self, expert_idx: int) -> Path:
        return self.cache_dir / f"proj_mat_layer_{self.layer_idx}_expert_{expert_idx}.pkl"
    def get_unified_projection_matrix(self) -> Optional[torch.Tensor]:
        return self._load_projection_matrix(-1)
    def _save_unified_projection_matrix(self, matrix: torch.Tensor):
        self._save_projection_matrix(-1, matrix)
    

def compute_projection_matrices_parallel(
    managers: List[ProjectionMatrixManager],
    num_samples: int = 100000,
    batch_size: int = 20,
    dataset_name: str = "wikipedia",
    force_recompute: bool = False
):
    active_items = []
    all_collectors = []

    print(f"Checking cache for {len(managers)} layers...")
    
    for mgr in managers:
        expert_indices = list(range(mgr.num_experts))
        to_compute = mgr._get_experts_to_compute(expert_indices, force_recompute)
        
        if to_compute:
            collector = mgr._prepare_collector(to_compute, store_on_cpu=True)
            active_items.append((mgr, collector, to_compute))
            all_collectors.append(collector)
        else:
            print(f"Layer {mgr.layer_idx}: All matrices cached.")

    if not active_items:
        print("All layers fully cached. Nothing to compute.")
        return

    ref_mgr = active_items[0][0] 
    
    print(f" Starting parallel collection for {len(active_items)} layers using {num_samples} samples...")
    
    ds = ref_mgr.get_dataset(dataset_name, num_samples)
    token_limit = 1024 * batch_size
    loader = DataLoader(
        ds, 
        batch_size=batch_size, 
        collate_fn=length_collation(token_limit),
        pin_memory=True
    )

    ref_mgr.model.eval()
    
    try:
        with torch.no_grad():
            for i, batch_list in tqdm(enumerate(loader), total=len(loader), desc="Collecting Stats"):
                for batch in batch_list:
                    batch = {k: v.to(ref_mgr.device) for k, v in batch.items()}
                    
                    for col in all_collectors:
                        col.set_current_batch_mask(batch['attention_mask'])
                    
                    ref_mgr.model(**batch)
                
                if i % 5 == 0: 
                    torch.cuda.empty_cache()
    finally:
        for col in all_collectors:
            col.cleanup()

    print("Computing SVDs and saving matrices...")
    for mgr, collector, indices in tqdm(active_items, desc="Finalizing Layers"):
        mgr._finalize_computation(collector, indices)
        