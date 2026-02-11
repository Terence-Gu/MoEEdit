# config.py
from dataclasses import dataclass


@dataclass
class MoEEditConfig:
    """Unified configuration for MoE Knowledge Editing"""
    # Model settings
    model_path: str = "path/to/model"
    device: str = "cuda"

    # Editor settings
    layer_idx: int = 13  # For backward compatibility with single-layer editing
    layers: list = None  # e.g., [11, 12, 13]
    lambda_reg: float = 0.15
    projection_threshold: float = 0.02
    nullspace_threshold: float = 0.02
    stats_dir: str = "./qwen3_moe_stats"
    # Caches
    target_cache_dir: str = "./target_vector_cache" # Used in edit number ablation.
    projection_cache_dir: str = "./projection_matrix_cache"  # Directory to cache per-expert projection matrices
    # Dataset name for target vector cache partitioning (e.g., 'zsre', 'counterfact')
    dataset_name: str | None = None
    num_layers: int = 48
    num_experts: int = 128
    num_experts_per_tok: int = 8
    d_model: int = 2048
    d_hidden: int = 768
    target_layer_for_unified_vector: int = None  
    layer_update_weights: dict = None  
    use_identity: bool = False
    num_samples: int = 1000
    use_expert_projection: bool = True
    streaming_covariance: bool = True
    method: str = "bcd"
    num_passes: int = 5
    learning_rate: float = 0.0001
    verify_targets: bool = False
    eval_batch_size: int = 4
    compare_pre_post: bool = True

    fact_token: str = "subject_last"  
    v_lr: float = 1e-1                  
    v_num_grad_steps: int = 50             
    v_loss_layer: int = -1                
    v_weight_decay: float = 1e-3          
    kl_factor: float = 0.0625            
    clamp_norm_factor: float = 4.0                  

    def get_editing_layers(self):
        """Get list of layers to edit"""
        if self.layers:
            return self.layers
        else:
            return [self.layer_idx]

    def get_target_layer_for_vector_computation(self):
        """Get the layer to use for unified target vector computation"""
        if self.is_multi_layer_enabled():
            if self.target_layer_for_unified_vector is not None:
                return self.target_layer_for_unified_vector
            else:
                # Default to the highest layer in editing layers
                editing_layers = self.get_editing_layers()
                return max(editing_layers)
        else:
            return self.layer_idx

    def is_multi_layer_enabled(self):
        """Check if multi-layer editing is enabled"""
        return self.layers is not None and len(self.layers) > 1

    @classmethod
    def from_json(cls, json_path: str):
        """Load configuration from JSON file"""
        import json
        with open(json_path, 'r') as f:
            data = json.load(f)

        config = cls()
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config







