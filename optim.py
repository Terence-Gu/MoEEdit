"""
Optimization algorithms for MoE knowledge editing
"""

import torch
from typing import List, Dict, Any
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from itertools import chain

class BlockCoordinateDescent:
    """Block Coordinate Descent optimizer for MoE down_proj weights with historical data support"""

    def __init__(self, num_experts: int, d_model: int, d_k: int, device: str, lambda_reg: float = 1e-2):
        self.num_experts = num_experts
        self.d_model = d_model
        self.d_k = d_k
        self.device = device
        self.lambda_reg = lambda_reg

        # Historical data storage for cumulative editing
        self.historical_data = {
            'gating': [],           # List of gating tensors from previous edits
            'keys': [],             # List of keys tensors from previous edits
            'values': [],           # List of values tensors from previous edits
            'residuals': [],        # List of residual tensors from previous edits
            'is_new': [],           # List of is_new flags from previous edits
            'entity_mapping': [],   # List of entity mappings from previous edits
            'prefix_mapping': []    
        }

        print(f"Experts: {num_experts}, d_model: {d_model}, d_k: {d_k}, λ: {lambda_reg}")

    def get_combined_data(self, current_stats: Dict) -> Dict:
        """
            Combine historical data with current editing data
        """
        if not self.historical_data['gating']:
            return current_stats

        combined_gating = torch.cat(self.historical_data['gating'] + [current_stats['gating']], dim=0)
        all_keys = list(chain.from_iterable(self.historical_data['keys'])) + current_stats['keys']
        all_values = list(chain.from_iterable(self.historical_data['values'])) + current_stats['values']
        all_residuals = list(chain.from_iterable(self.historical_data['residuals'])) + current_stats['residuals']
        all_is_new = list(chain.from_iterable(self.historical_data['is_new'])) + current_stats['is_new']
        all_entity_mapping = list(chain.from_iterable(self.historical_data['entity_mapping'])) + current_stats['entity_mapping']
        all_prefix_mapping = list(chain.from_iterable(self.historical_data['prefix_mapping'])) + current_stats['prefix_mapping']

        combined_stats = {
            'gating': combined_gating,
            'keys': all_keys,
            'values': all_values,
            'residuals': all_residuals,
            'is_new': all_is_new,
            'entity_mapping': all_entity_mapping,
            'prefix_mapping': all_prefix_mapping
        }

        print(f"Combined data: {len(all_residuals)} total samples ({len(current_stats['residuals'])} new + {len(all_residuals) - len(current_stats['residuals'])} historical)")
        return combined_stats
  
    def optimize(self, stats: Dict, num_passes: int = 2, projection_matrices: Dict = None, seed: int = 42) -> List[torch.Tensor]:
        """Random-Permutation BCD (RP-BCD) optimization with optional historical data support.

        Args:
            stats: Current batch statistics (keys, values, gating, residuals).
            num_passes: Number of BCD passes over the data.
            projection_matrices: Optional expert_id -> projection matrix P_j mapping.
            seed: Random seed
        """

        working_stats = self.get_combined_data(stats)
        print(f"Running RP-BCD optimization with {num_passes} passes ...")

        # Prepare tensors 
        residuals_batch = working_stats['residuals']
        R = torch.stack([r.clone().float() for r in residuals_batch])
        T = len(residuals_batch)
        gating = working_stats['gating'].float()

        keys_tensor = torch.zeros(T, self.num_experts, self.d_k, device=self.device, dtype=torch.float32)
        values_tensor = torch.zeros(T, self.num_experts, self.d_k, device=self.device, dtype=torch.float32)
        valid_mask = torch.zeros(T, self.num_experts, device=self.device, dtype=torch.bool)

        for t in range(T):
            for j in range(self.num_experts):
                if working_stats['keys'][t][j] is not None:
                    key_tj = working_stats['keys'][t][j].float()
                    keys_tensor[t, j] = key_tj

                    if working_stats['values'][t][j] is not None:
                        values_tensor[t, j] = working_stats['values'][t][j].float()
                    elif projection_matrices and j in projection_matrices:
                        P_j = projection_matrices[j].to(self.device)
                        values_tensor[t, j] = P_j @ key_tj
                    else:
                        values_tensor[t, j] = key_tj
                    valid_mask[t, j] = True

        print(f"RP-BCD on {T} samples, {self.num_experts} experts.")

        deltas = [torch.zeros(self.d_model, self.d_k, device=self.device, dtype=torch.float32)
                for _ in range(self.num_experts)]

        g = torch.Generator(device='cpu')
        g.manual_seed(seed)

        for pass_idx in range(num_passes):
            perm = torch.randperm(self.num_experts, generator=g)
            # perm = list(range(self.num_experts))

            for j in perm:
                j = j.item()
                active_mask = (gating[:, j] > 0) & valid_mask[:, j]
                if not active_mask.any():
                    continue

                S_j = R.clone()
                for ell in range(self.num_experts):
                    if ell == j:
                        continue
                    contrib = gating[:, ell:ell+1] * (values_tensor[:, ell, :] @ deltas[ell].T)
                    S_j += contrib

                j_indices = torch.where(active_mask)[0]
                g_j = gating[j_indices, j]
                v_j = values_tensor[j_indices, j]
                S_j_active = S_j[j_indices]

                weighted_vs = g_j.unsqueeze(1) * v_j
                B_j = S_j_active.T @ weighted_vs
                weights = g_j ** 2
                M_j = v_j.T @ (weights.unsqueeze(1) * v_j)
                reg_term = self.lambda_reg * torch.eye(self.d_k, device=self.device, dtype=torch.float32)
                M_j_total = M_j + reg_term

                try:
                    deltas[j] = -B_j @ torch.linalg.inv(M_j_total)
                except RuntimeError:
                    deltas[j] = -B_j @ torch.linalg.pinv(M_j_total) # If inv fails, try pseudo-inv

        print("RP-BCD optimization completed.")
        return deltas