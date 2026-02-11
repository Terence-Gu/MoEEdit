#!/usr/bin/env python3
# evaluate.py
"""
MoE Knowledge Editing Evaluation Script
"""

import os
import sys
import json
import time
import shutil
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from .editor import MoEKnowledgeEditor
from .utils import EditRequest, get_tqdm_kwargs
from .config import MoEEditConfig
from .manager import Qwen3MoEStatisticsCollector


# Path setup
root_dir = os.path.join(os.path.dirname(__file__), '..')
abs_root_dir = os.path.abspath(root_dir)
if abs_root_dir not in sys.path:
    sys.path.insert(0, abs_root_dir)
from moe_edit.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
from moe_edit.py.eval_utils_zsre import compute_rewrite_quality_zsre
from dsets import AttributeSnippets, get_tfidf_vectorizer



class ExpertActivationAnalyzer:
    """Analyzes changes in expert activation distributions before and after updates."""

    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.num_experts_per_tok = getattr(model.config, 'num_experts_per_tok', 8)

    def collect_expert_activations(self, requests: List[EditRequest], layer_indices: List[int]) -> Dict[int, Dict[str, Any]]:
        """Collects expert activation distribution (mean probs + top-k indices)."""
        
        activations = {l: {'mean_probs': [], 'indices': []} for l in layer_indices}
        batch_size = 50
        
        for i in tqdm(range(0, len(requests), batch_size), desc="Processing batches"):
            batch_reqs = requests[i : i + batch_size]
            batch_data = self._collect_batch(batch_reqs, layer_indices)
            
            for l in layer_indices:
                if l in batch_data:
                    activations[l]['mean_probs'].extend(batch_data[l]['mean_probs'])
                    activations[l]['indices'].extend(batch_data[l]['indices'])

        for l in layer_indices:
            activations[l]['mean_probs'] = np.array(activations[l]['mean_probs'])

        return activations

    def _collect_batch(self, requests: List[EditRequest], layer_indices: List[int]) -> Dict[int, Dict[str, Any]]:
        results = {l: {'mean_probs': [], 'indices': []} for l in layer_indices}
        
        for req in requests:
            inputs = self.tokenizer(req.prompt, return_tensors="pt").to(self.device)
            sample_data = self._forward_capture(inputs, layer_indices)
            
            for l in layer_indices:
                if l in sample_data:
                    results[l]['mean_probs'].append(sample_data[l]['mean_probs'])
                    results[l]['indices'].append(sample_data[l]['indices'])
                else:
                    raise ValueError(f"Layer {l} not found in sample data")
        return results

    def _forward_capture(self, inputs, layer_indices: List[int]) -> Dict[int, Dict[str, Any]]:
        results = {}
        collector = Qwen3MoEStatisticsCollector()
        
        hooks = []
        for layer in self.model.model.layers:
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
                hooks.extend(collector.create_hooks(layer.mlp))

        try:
            with torch.no_grad():
                self.model(**inputs)

            for router_idx, router_logit in enumerate(collector.router_logits):
                if router_idx in layer_indices:
                    logits = router_logit[0] if router_logit.dim() == 3 else router_logit
                    
                    if logits.numel() > 0:
                        current_step_logits = logits[-1]
                        _, top_k_indices = torch.topk(current_step_logits.float(), self.num_experts_per_tok, dim=-1)
                        probs = torch.softmax(current_step_logits.float(), dim=-1)
                        
                        results[router_idx] = {
                            'mean_probs': probs.cpu().numpy(),
                            'indices': top_k_indices.unsqueeze(0).cpu().numpy().astype(np.int16)
                        }
        except Exception as e:
            print(f"Activation capture failed: {e}")
        finally:
            for h in hooks: h.remove()
            collector.clear()

        return results
    def analyze_distribution_changes(self, pre_data: Dict, post_data: Dict) -> Dict[str, Any]:
        results = {}
        for l in pre_data.keys():
            if l not in post_data: continue
            results[l] = self._analyze_layer(
                pre_data[l]['mean_probs'], post_data[l]['mean_probs'],
                pre_data[l]['indices'], post_data[l]['indices'], l
            )
        return results

    def _analyze_layer(self, pre_mean, post_mean, pre_indices, post_indices, layer_idx):
        pre_g = np.mean(pre_mean, axis=0)
        post_g = np.mean(post_mean, axis=0)
        diff = post_g - pre_g

        pre_m = self._calc_heavy_tail(pre_g)
        post_m = self._calc_heavy_tail(post_g)
        kl = self._safe_kl(post_g, pre_g)
        m = 0.5 * (post_g + pre_g)
        js = 0.5 * self._safe_kl(post_g, m) + 0.5 * self._safe_kl(pre_g, m)
        rs_samples = []
        for pre_seq, post_seq in zip(pre_indices, post_indices):
            min_len = min(len(pre_seq), len(post_seq))
            if min_len == 0: continue
            
            p1, p2 = pre_seq[:min_len], post_seq[:min_len]
            matches = (p1[:, :, None] == p2[:, None, :])
            intersect = matches.any(axis=2).sum(axis=1)
            union = (2 * self.num_experts_per_tok) - intersect
            rs_samples.append(np.mean(intersect / union))
            
        rs_top8 = float(np.mean(rs_samples)) if rs_samples else 0.0

        return {
            "layer_idx": layer_idx,
            "pre_metrics": pre_m, "post_metrics": post_m,
            "kl_divergence": float(kl), "js_divergence": float(js), "rs_top8": rs_top8,
            "top_k": int(self.num_experts_per_tok),
            "max_increase_expert": int(np.argmax(diff)), "max_increase_value": float(np.max(diff)),
            "max_decrease_expert": int(np.argmin(diff)), "max_decrease_value": float(np.min(diff)),
            "mean_absolute_change": float(np.mean(np.abs(diff))),
            "total_variation_distance": float(0.5 * np.sum(np.abs(diff)))
        }

    @staticmethod
    def _calc_heavy_tail(dist):
        sorted_dist = np.sort(dist)[::-1]
        total = np.sum(sorted_dist) or 1e-10
        n = len(sorted_dist)
        gini = (2 * np.sum(np.arange(1, n+1) * sorted_dist)) / (n * total) - (n + 1) / n
        return {
            "gini": gini,
            "top_10_ratio": np.sum(sorted_dist[:max(1, int(0.1*n))]) / total,
            "top_20_ratio": np.sum(sorted_dist[:max(1, int(0.2*n))]) / total
        }

    @staticmethod
    def _safe_kl(p, q, eps=1e-10):
        p, q = p + eps, q + eps
        return np.sum((p/p.sum()) * np.log((p/p.sum()) / (q/q.sum())))

    def visualize_activation_changes(self, pre, post, results, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        # Save JSON
        with open(os.path.join(out_dir, 'expert_activation_analysis.json'), 'w') as f:
            json.dump(self._convert_numpy(results), f, indent=2)
        # Save Summary Text
        with open(os.path.join(out_dir, 'analysis_summary.txt'), 'w') as f:
            f.write(self._create_summary(results))
        print(f"Expert activation analysis saved to {out_dir}")

    def _convert_numpy(self, obj):
        if isinstance(obj, dict): return {k: self._convert_numpy(v) for k, v in obj.items()}
        if isinstance(obj, list): return [self._convert_numpy(i) for i in obj]
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)): return float(obj)
        if isinstance(obj, (np.int32, np.int64)): return int(obj)
        if isinstance(obj, np.bool_): return bool(obj)
        return obj

    def _create_summary(self, results: Dict[str, Any]) -> str:
        lines = ["="*80, "EXPERT ACTIVATION DISTRIBUTION ANALYSIS SUMMARY", "="*80, ""]
        layer_indices = sorted(results.keys())

        for l in layer_indices:
            r = results[l]
            lines.extend([
                f"Layer {l}:", "-"*40,
                f"  Gini: {r['pre_metrics']['gini']:.4f} → {r['post_metrics']['gini']:.4f}",
                f"  Top10%: {r['pre_metrics']['top_10_ratio']:.4f} → {r['post_metrics']['top_10_ratio']:.4f}",
                f"  KL: {r['kl_divergence']:.6f}, JS: {r['js_divergence']:.6f}",
                f"  RS (Top-8): {r.get('rs_top8', 0):.4f}",
                f"  Max Inc: Exp {r['max_increase_expert']} ({r['max_increase_value']:.4f})",
                f"  Max Dec: Exp {r['max_decrease_expert']} ({r['max_decrease_value']:.4f})",
                ""
            ])

        # Overall
        avg_js = np.mean([results[l]['js_divergence'] for l in layer_indices])
        avg_rs = np.nanmean([results[l].get('rs_top8', np.nan) for l in layer_indices])
        impact = "HIGH" if avg_js > 0.1 else "MEDIUM" if avg_js > 0.01 else "LOW"

        lines.extend([
            "OVERALL SUMMARY:", "-"*40,
            f"Average JS: {avg_js:.6f}",
            f"Average RS: {avg_rs:.4f}",
            f"Overall Impact: {impact}",
            "", "="*80
        ])
        return "\n".join(lines)


class MoEEditHyperParams:
    """Hyperparameters wrapper"""
    def __init__(self, **kwargs):
        self.model_name = kwargs.get("model_name", "Qwen3-30B-A3B")
        self.layers = kwargs.get("layers", [13])
        self.layer_selection = kwargs.get("layer_selection", "all")
        self.fact_token = kwargs.get("fact_token", "subject_last")
        self.lambda_reg = kwargs.get("lambda_reg", 0.15)
        self.projection_threshold = kwargs.get("projection_threshold", 0.02)
        self.num_experts = kwargs.get("num_experts", 128)
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 8)
        self.d_model = kwargs.get("d_model", 2048)
        self.d_hidden = kwargs.get("d_hidden", 768)
        self.optimization_method = kwargs.get("optimization_method", "bcd")
        self.num_passes = kwargs.get("num_passes", 5)
        self.num_steps = kwargs.get("num_steps", 250)
        self.learning_rate = kwargs.get("learning_rate", 0.0001)
        self.verify_targets = kwargs.get("verify_targets", False)
        self.use_identity = kwargs.get("use_identity", False)
        self.unified = kwargs.get("unified", False)
        self.num_samples = kwargs.get("num_samples", 10000)
        self.use_expert_projection = kwargs.get("use_expert_projection", False)
        self.mom2_dataset = kwargs.get("mom2_dataset", "wikipedia")
        self.stats_dir = kwargs.get("stats_dir", "./qwen3_moe_stats")
        self.target_cache_dir = kwargs.get("target_cache_dir", "./target_vector_cache")
        self.projection_cache_dir = kwargs.get("projection_cache_dir", "./projection_matrix_cache")
        self.dataset_name = kwargs.get("dataset_name", None)
        self.batch_size = kwargs.get("batch_size", 4)
        self.compare_pre_post = kwargs.get("compare_pre_post", True)
        self.enable_text_generation = kwargs.get("enable_text_generation", True)
        self.enable_expert_activation_analysis = kwargs.get("enable_expert_activation_analysis", True)
        self.v_lr = kwargs.get("v_lr", 0.1)
        self.v_num_grad_steps = kwargs.get("v_num_grad_steps", 25)
        self.v_loss_layer = kwargs.get("v_loss_layer", 47)
        self.v_weight_decay = kwargs.get("v_weight_decay", 0.5)
        self.kl_factor = kwargs.get("kl_factor", 0.0625)
        self.clamp_norm_factor = kwargs.get("clamp_norm_factor", 0.75)
        self.auto_infer_dims = kwargs.get("auto_infer_dims", True)
        self.dataset_size_limit = kwargs.get("dataset_size_limit", None)
        self.target_layer_for_unified_vector = kwargs.get("target_layer_for_unified_vector", None)

    @classmethod
    def from_json(cls, json_path: str):
        with open(json_path, 'r') as f: return cls(**json.load(f))

REMOTE_ROOT_URL = "https://rome.baulab.info/data/dsets"
DATASET_URLS = {
    "counterfact": f"{REMOTE_ROOT_URL}/counterfact.json",
    "multi_counterfact": f"{REMOTE_ROOT_URL}/multi_counterfact.json",
    "zsre": f"{REMOTE_ROOT_URL}/zsre_mend_eval.json"
}

class DatasetLoader:
    """Consolidated dataset loader with auto-download capability."""
    @staticmethod
    def _ensure_file(data_dir: str, filename: str, url_key: str) -> Path:
        """Check if file exists, otherwise download it."""
        path = Path(data_dir) / filename
        if not path.exists():
            if url_key not in DATASET_URLS:
                raise ValueError(f"No remote URL defined for {url_key}")
            
            url = DATASET_URLS[url_key]
            print(f"Dataset {filename} not found. Downloading from {url}...")
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                torch.hub.download_url_to_file(url, str(path))
            except Exception as e:
                if path.exists(): path.unlink() 
                raise RuntimeError(f"Failed to download dataset: {e}")
        return path

    @staticmethod
    def load_counterfact(data_dir: str, limit: Optional[int] = None) -> List[Dict]:
        path = DatasetLoader._ensure_file(data_dir, "counterfact.json", "counterfact")
        with open(path, 'r') as f: 
            data = json.load(f)
        return data[:limit] if limit else data

    @staticmethod
    def load_multi_counterfact(data_dir: str, limit: Optional[int] = None) -> List[Dict]:
        path = DatasetLoader._ensure_file(data_dir, "multi_counterfact.json", "multi_counterfact")
        with open(path, 'r') as f: 
            data = json.load(f)
        return data[:limit] if limit else data

    @staticmethod
    def load_zsre(data_dir: str, tok, limit: Optional[int] = None) -> List[Dict]:
        path = DatasetLoader._ensure_file(data_dir, "zsre_mend_eval.json", "zsre")
        
        with open(path, 'r') as f:
            raw = json.load(f)
            
        data = []
        for i, r in enumerate(raw):
            if "nq question: " not in r["loc"]:
                continue 
                
            ans_toks = tok(" " + r["loc_ans"])["input_ids"]
            data.append({
                "case_id": i,
                "requested_rewrite": {
                    "prompt": r["src"].replace(r["subject"], "{}"),
                    "subject": r["subject"],
                    "target_new": {"str": r["answers"][0]},
                    "target_true": {"str": "<|endoftext|>"},
                },
                "paraphrase_prompts": [r["rephrase"]],
                "neighborhood_prompts": [{"prompt": r["loc"] + "?" + tok.decode(ans_toks[:j]), "target": tok.decode(ans_toks[j])} for j in range(len(ans_toks))],
                "attribute_prompts": [], "generation_prompts": [],
            })
        return data[:limit] if limit else data

    @staticmethod
    def load_dataset(name: str, data_dir: str, tok=None, limit: int = None):
        if name in ["counterfact", "cf"]: return DatasetLoader.load_counterfact(data_dir, limit)
        if name in ["multi_counterfact", "mcf"]: return DatasetLoader.load_multi_counterfact(data_dir, limit)
        if name == "zsre": return DatasetLoader.load_zsre(data_dir, tok, limit)
        raise ValueError(f"Unknown dataset: {name}")

    @staticmethod
    def load_activation_analysis_samples(name: str, data_dir: str, limit: int = None, tok=None):
        full = DatasetLoader.load_dataset(name, data_dir, tok, None)
        limit = limit or (len(full) // 2)
        total_needed = limit * 2
        
        if len(full) < total_needed:
            print(f"Dataset has only {len(full)} samples, need {total_needed}. Splitting what we have.")
            mid = len(full) // 2
            return {'editing': full[:mid], 'comparison': full[mid:]}
        
        return {'editing': full[:limit], 'comparison': full[limit:limit*2]}

class MoEEditEvaluator:
    """Main evaluator for MoE Knowledge Editing"""

    def __init__(self, hparams: MoEEditHyperParams, model_path: str, device="cuda", force_recompute_projection=False):
        self.hparams = hparams
        self.model_path = model_path
        self.device = device
        self.force_recompute_projection = force_recompute_projection

        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token

        # Auto-infer dims
        if self.hparams.auto_infer_dims:
            try:
                self.hparams.d_model = getattr(self.model.config, 'hidden_size', self.hparams.d_model)
                tl = self.model.model.layers[self.hparams.layers[0]].mlp
                if hasattr(tl, 'experts'):
                    dp = getattr(tl.experts, 'down_proj', None)
                    if dp is not None and hasattr(dp, 'shape') and len(dp.shape) == 3:
                        self.hparams.num_experts = dp.shape[0]
                        expert_dim = dp.shape[1]
                        self.hparams.d_hidden = expert_dim
            except Exception as e:
                print(f"Auto-infer dims failed: {e}")

        self.editor = self._create_editor()
        self.activation_analyzer = ExpertActivationAnalyzer(self.model, self.tokenizer, self.device)

    def _create_editor(self):
        config = MoEEditConfig(
            layer_idx=self.hparams.layers[0],
            layers=self.hparams.layers,
            target_layer_for_unified_vector=self.hparams.target_layer_for_unified_vector,
            num_experts_per_tok=self.hparams.num_experts_per_tok,
            lambda_reg=self.hparams.lambda_reg,
            projection_threshold=self.hparams.projection_threshold,
            verify_targets=self.hparams.verify_targets,
            use_identity=self.hparams.use_identity,
            method=self.hparams.optimization_method,
            num_passes=self.hparams.num_passes,
            learning_rate=self.hparams.learning_rate,
            num_experts=self.hparams.num_experts,
            d_model=self.hparams.d_model,
            d_hidden=self.hparams.d_hidden,
            stats_dir=self.hparams.stats_dir,
            target_cache_dir=self.hparams.target_cache_dir,
            projection_cache_dir=self.hparams.projection_cache_dir,
            dataset_name=getattr(self.hparams, 'dataset_name', None),
            fact_token=getattr(self.hparams, 'fact_token', 'subject_last'),
            num_samples=self.hparams.num_samples,
            model_path=self.model_path
        )
        for k in ['v_lr', 'v_num_grad_steps', 'v_loss_layer', 'v_weight_decay', 'kl_factor', 'clamp_norm_factor']:
            setattr(config, k, getattr(self.hparams, k))

        print(f" Creating Editor for layers {self.hparams.layers} (Multi-layer: {len(self.hparams.layers)>1})")
        return MoEKnowledgeEditor(self.model, self.tokenizer, config)

    def _initialize_detailed_eval_components(self):
        d = Path(__file__).parent / "data"
        return AttributeSnippets(str(d)), get_tfidf_vectorizer(str(d))

    def convert_to_edit_requests(self, dataset: List[Dict]) -> List[EditRequest]:
        reqs = []
        for i, item in enumerate(dataset):
            case_id = item.get("case_id", i)
            if "requested_rewrite" in item:
                rw = item["requested_rewrite"]
                p, s, t = rw["prompt"], rw["subject"], rw["target_new"]["str"]
            else:
                p = item.get("prompt", item.get("src", ""))
                s = item.get("subject", "")
                t = item.get("target_new", item.get("target", item.get("answers", [])))
                if isinstance(t, dict): t = t["str"]
                elif isinstance(t, list) and t: t = t[0]

            if p.count("{}") > 1: continue
            if not t.startswith(" "): t = " " + t
            reqs.append(EditRequest(case_id=case_id, prompt=p, target_new=t, subject=s))
        print(f"Converted {len(reqs)} requests")
        return reqs

    def evaluate_dataset(self, dataset: List[Dict], output_dir: str = None, activation_samples=None, dataset_name=None):
        start_time = time.time()
        self._dataset_name = dataset_name
        snips, vec = self._initialize_detailed_eval_components()
        
        requests = self.convert_to_edit_requests(dataset)
        batch_size = self.hparams.batch_size
        self._current_output_dir = output_dir
        if output_dir: Path(output_dir).mkdir(parents=True, exist_ok=True)

        pre_acts = {'editing': None, 'comparison': None}
        an_reqs = {'editing': None, 'comparison': None}
        an_layers = self._get_analysis_layers()
        
        if self.hparams.enable_expert_activation_analysis:
            # Setup requests
            if activation_samples:
                an_reqs['editing'] = self.convert_to_edit_requests(activation_samples['editing'])
                an_reqs['comparison'] = self.convert_to_edit_requests(activation_samples['comparison'])
            else:
                an_reqs['editing'] = requests


            pre_acts['editing'] = self.activation_analyzer.collect_expert_activations(an_reqs['editing'], an_layers)
            if an_reqs['comparison']:
                pre_acts['comparison'] = self.activation_analyzer.collect_expert_activations(an_reqs['comparison'], an_layers)

        print("Applying edits...")
        total_edits = 0
        tqdm_args = get_tqdm_kwargs(desc="Applying cumulative edits")
        
        for i in tqdm(range(0, len(requests), batch_size), **tqdm_args):
            batch = requests[i:i + batch_size]
            print(f"Editing batch {i//batch_size + 1} ({len(batch)} reqs), total so far: {total_edits}")
            self.apply_batch_edits(batch, i//batch_size + 1, total_edits)
            total_edits += len(batch)

        if self.hparams.enable_expert_activation_analysis and pre_acts['editing']:
            self._analyze_and_report(an_reqs['editing'], pre_acts['editing'], an_layers, "editing_samples")
            if pre_acts['comparison']:
                self._analyze_and_report(an_reqs['comparison'], pre_acts['comparison'], an_layers, "comparison_samples")

        print("Evaluating...")
        all_results = []
        tqdm_args = get_tqdm_kwargs(desc="Evaluating requests")
        
        for req in tqdm(requests, **tqdm_args):
            orig = next((r for r in dataset if r.get("case_id") == req.case_id), dataset[req.case_id] if req.case_id < len(dataset) else None)
            res = self.evaluate_single_request(req, orig, total_edits, snips, vec)
            all_results.append(res)
            
            if output_dir:
                with open(Path(output_dir) / f"{res['num_edits']}_edits-case_{res['case_id']}.json", 'w') as f:
                    json.dump(res, f, indent=1, default=str)

        # Summary
        success = [r for r in all_results if r.get("post", {}).get("rewrite_prompts_correct", [False])[0]]
        summary = {
            "total_samples": len(all_results), "successful_edits": len(success),
            "success_rate": len(success) / len(all_results) if all_results else 0,
            "total_cumulative_edits": total_edits,
            "timing": {"total_time": time.time() - start_time},
            "results": all_results
        }
        
        if output_dir:
            with open(Path(output_dir) / "evaluation_summary.json", 'w') as f: json.dump(summary, f, indent=2, default=str)
        
        print(f"Eval complete. Success: {summary['success_rate']:.2%}")
        return summary

    def _get_analysis_layers(self):
        """
        Return all layer indices for analysis if enabled.
        """
        if hasattr(self.model.config, "num_hidden_layers"):
            num_layers = self.model.config.num_hidden_layers
        elif hasattr(self.model.config, "num_layers"):
            num_layers = self.model.config.num_layers
        elif hasattr(self.model.config, "n_layer"):
            num_layers = self.model.config.n_layer
        else:
            num_layers = len(self.model.model.layers)
            
        return list(range(num_layers))

    def _analyze_and_report(self, reqs, pre_acts, layers, tag):

        post_acts = self.activation_analyzer.collect_expert_activations(reqs, layers)
        results = self.activation_analyzer.analyze_distribution_changes(pre_acts, post_acts)
        # self._log_analysis_summary(results)
        
        if self._current_output_dir:
            out = os.path.join(self._current_output_dir, "expert_activation_analysis", tag)
            self.activation_analyzer.visualize_activation_changes(pre_acts, post_acts, results, out)

    def apply_batch_edits(self, batch, batch_num, total_edits):
        orig_lambdas = {}
        if total_edits > 0:
            prog_lambda = self.hparams.lambda_reg * (1.0 + total_edits / 1000.0)
            for l, data in self.editor.layer_data.items():
                orig_lambdas[l] = data['bcd_optimizer'].lambda_reg
                data['bcd_optimizer'].lambda_reg = prog_lambda

        self.editor.edit_knowledge(batch, self.hparams.optimization_method, self.hparams.num_passes, self.hparams.verify_targets)

        if orig_lambdas:
            for l, val in orig_lambdas.items(): self.editor.layer_data[l]['bcd_optimizer'].lambda_reg = val
                

    def evaluate_single_request(self, req: EditRequest, orig_record: Dict, total_edits: int, snips=None, vec=None):
        record = orig_record.copy() if orig_record else {
            "case_id": req.case_id,
            "requested_rewrite": {"prompt": req.prompt, "subject": req.subject, "target_new": {"str": req.target_new}, "target_true": {"str": ""}},
            "paraphrase_prompts": [], "neighborhood_prompts": [], "generation_prompts": []
        }
        
        if getattr(self, '_dataset_name', '') == 'zsre':
            metrics = compute_rewrite_quality_zsre(self.model, self.tokenizer, record, snips, vec)
        else:
            metrics = compute_rewrite_quality_counterfact(self.model, self.tokenizer, record, snips, vec, self.hparams.enable_text_generation)

        success = metrics.get("rewrite_prompts_correct", [False])[0]
        print(f"  Case {req.case_id}: {'✔️' if success else '✖️'} (edits: {total_edits})")
        
        return {    
            "case_id": req.case_id, "grouped_case_ids": [req.case_id], "num_edits": total_edits,
            "requested_rewrite": record["requested_rewrite"], "time": 0.0, "post": metrics
        }


    def _log_analysis_summary(self, results):
        print(f" Expert Activation Analysis Summary:")
        for l in sorted(results.keys()):
            r = results[l]
            print(f"  Layer {l}: Gini {r['pre_metrics']['gini']:.3f}->{r['post_metrics']['gini']:.3f}, JS {r['js_divergence']:.6f}, RS {r.get('rs_top8', 0):.4f}")
        avg_js = np.mean([results[l]['js_divergence'] for l in results])


def generate_run_name(out_dir: str) -> str:
    path = Path(out_dir) / "MoEEdit"
    if not path.exists(): return "run_000"
    ids = [int(p.name.split("_")[-1]) for p in path.iterdir() if p.is_dir() and p.name.startswith("run_") and p.name.split("_")[-1].isdigit()]
    return f"run_{str(max(ids) + 1 if ids else 0).zfill(3)}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hparams", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset", required=True, choices=["counterfact", "cf", "multi_counterfact", "mcf", "zsre"])
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--force_recompute_projection", action="store_true")
    args = parser.parse_args()

    hparams = MoEEditHyperParams.from_json(args.hparams)
    hparams.dataset_name = args.dataset
    
    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    samples = DatasetLoader.load_activation_analysis_samples(args.dataset, args.data_dir, args.limit, tok)
    
    out_dir = None
    if args.output_dir:
        run_name = args.run_name or generate_run_name(args.output_dir)
        out_dir = Path(args.output_dir) / "MoEEdit" / run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.hparams, out_dir / "params.json")
        with open(out_dir / "run_config.json", 'w') as f: 
            json.dump({"args": vars(args), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
        print(f"Results: {out_dir}")

    evaluator = MoEEditEvaluator(hparams, args.model_path, args.device, args.force_recompute_projection)
    evaluator.evaluate_dataset(samples['editing'], str(out_dir) if out_dir else None, samples, args.dataset)

if __name__ == "__main__":
    main()