import torch
import numpy as np
import json
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple
from pathlib import Path

# Global variables (set during initialization)
DEVICE = "cuda"
model = None
tokenizer = None
LAYER_MAP = None

#1. Layer selection
def select_layers(model):
    n_layers = model.config.num_hidden_layers
    def at_fraction(numer, denom):
        return min(n_layers - 1, max(0, int(n_layers * numer / denom)))

    layer_map = {
        "first": 0,
        "last": n_layers - 1,
        "1/4": at_fraction(1, 4),
        "2/4": at_fraction(2, 4),
        "1/5": at_fraction(1, 5),
        "2/5": at_fraction(2, 5),
        "3/5": at_fraction(3, 5),
        "4/5": at_fraction(4, 5),
        "early": int(n_layers * 0.25),
        "mid": int(n_layers * 0.5),
        "late": n_layers - 1,
    }
    # Per-layer addressable keys: e.g. "L0", "L1", ..., and bare "0", "1", ...
    # Lets `--rep_stiff_layers` probe arbitrary individual transformer layers.
    for i in range(n_layers):
        layer_map[f"L{i}"] = i
        layer_map[str(i)] = i
    return layer_map

#2. Model setup
def initialize_model(model_name: str):
    """Initialize model, tokenizer, and compute layer map."""
    global model, tokenizer, LAYER_MAP
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        output_hidden_states=True,
        torch_dtype=torch.float16,
    ).to(DEVICE)
    model.eval()
    
    # Compute layer map
    LAYER_MAP = select_layers(model)
    
    return model, tokenizer, LAYER_MAP

#3. Load data files
def load_similar_questions(file_path: str) -> List[Dict]:
    """Load similar_questions.json file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_incomplete_questions(file_path: str) -> Dict[Tuple[int, int], Dict]:
    """
    Load incomplete_questions.jsonl and create a mapping.
    Key: (original_question_id, similar_question_id or None)
    Value: dict with 'complete' and 'incomplete' question texts
    """
    mapping = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                orig_id = entry.get("original_question_id")
                similar_id = entry.get("similar_question_id")
                
                # Use None for original questions, similar_id for similar questions
                key = (orig_id, similar_id if entry.get("type") == "similar" else None)
                mapping[key] = {
                    "complete": entry["original_question"],
                    "incomplete": entry["incomplete_question"],
                }
    return mapping

#4. Embedding extraction
@torch.no_grad()
def extract_representation(
    text: str,
    layer_idx: int,
) -> torch.Tensor:
    """
    Extract representation f_l(Q) for question Q at layer l.
    Uses mean-pooling over tokens (stable).
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=False,
        truncation=True,
    ).to(DEVICE)

    outputs = model(**inputs, output_hidden_states=True)
    # hidden_states[0] is the embedding output; layer_idx numbers transformer blocks.
    hidden = outputs.hidden_states[layer_idx + 1]  # (1, T, D)

    # Mean pooling over tokens
    rep = hidden.mean(dim=1).squeeze(0)  # (D,)
    return rep

#5. Metric 2: Directional Collapse (alignment with "template shift")
#utility: cosine similarity (safe)
def cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    """
    Compute cosine similarity between two vectors.
    """
    # IMPORTANT: do the reduction math in fp32.
    # In fp16, `dot` and `norm(a)*norm(b)` can overflow to inf; inf/inf => NaN.
    a32 = a.float()
    b32 = b.float()
    return (torch.dot(a32, b32) / (torch.norm(a32) * torch.norm(b32) + eps)).item()

#compute shift vector for one question
def compute_shift_vector(
    q_complete: str,
    q_incomplete: str,
    layer_idx: int,
) -> torch.Tensor:
    """
    Compute shift vector: f_l(Q^(g)) - f_l(Q^(-g))
    where Q^(g) is complete and Q^(-g) is incomplete.
    """
    f_complete = extract_representation(q_complete, layer_idx)
    f_incomplete = extract_representation(q_incomplete, layer_idx)
    return f_complete - f_incomplete

#directional collapse for one group
def compute_group_alignment(
    group_data: Dict,
    incomplete_mapping: Dict[Tuple[int, int], Dict],
    layer_idx: int,
):
    """
    Compute directional collapse metric for a group.
    
    Formula:
    1. Mean shift direction from similars: s_l^(g) = (1/K) * Σ_{i=1}^{K} (f_l(Q_i^(g)) - f_l(Q_i^(-g)))
    2. Alignment for original: Align_l^(g) = cos(f_l(Q_0^(g)) - f_l(Q_0^(-g)), s_l^(g))
    
    Returns:
        align: cosine alignment score for original
        template_shift_norm: norm of mean shift direction from similars
        original_shift_norm: norm of original shift vector
    """
    orig_question_id = group_data["original_question_id"]
    
    # Get original question's complete and incomplete versions
    orig_key = (orig_question_id, None)
    if orig_key not in incomplete_mapping:
        raise ValueError(f"Missing incomplete version for original question {orig_question_id}")
    
    orig_complete = incomplete_mapping[orig_key]["complete"]
    orig_incomplete = incomplete_mapping[orig_key]["incomplete"]
    
    # Compute shift vector for original: f_l(Q_0^(g)) - f_l(Q_0^(-g))
    original_shift = compute_shift_vector(orig_complete, orig_incomplete, layer_idx)
    
    # Compute shift vectors for similar questions (i = 1..K)
    sim_shifts = []
    for similar_q in group_data.get("similar_questions", []):
        similar_id = similar_q.get("id")
        similar_key = (orig_question_id, similar_id)
        
        if similar_key not in incomplete_mapping:
            continue  # Skip if incomplete version not found
        
        similar_complete = incomplete_mapping[similar_key]["complete"]
        similar_incomplete = incomplete_mapping[similar_key]["incomplete"]
        
        # Compute shift: f_l(Q_i^(g)) - f_l(Q_i^(-g))
        shift = compute_shift_vector(similar_complete, similar_incomplete, layer_idx)
        sim_shifts.append(shift)
    
    if len(sim_shifts) == 0:
        return {
            "alignment": np.nan,
            "template_shift_norm": np.nan,
            "original_shift_norm": torch.norm(original_shift).item(),
        }
    
    # Compute mean shift direction from similars: s_l^(g) = (1/K) * Σ_{i=1}^{K} shift_i
    sim_shifts_tensor = torch.stack(sim_shifts, dim=0)  # (K, D)
    template_shift = sim_shifts_tensor.mean(dim=0)  # (D,)
    
    # Compute alignment: Align_l^(g) = cos(original_shift, template_shift)
    align = cosine_sim(original_shift, template_shift)
    
    return {
        "alignment": align,
        "template_shift_norm": torch.norm(template_shift).item(),
        "original_shift_norm": torch.norm(original_shift).item(),
    }

#run across layers + groups
def run_alignment_experiment(
    similar_questions_file: str,
    incomplete_questions_file: str,
) -> List[Dict]:
    """
    Run the directional collapse experiment for all groups and layers.
    Returns results with alignment scores for each group and layer.
    """
    # Load data
    groups = load_similar_questions(similar_questions_file)
    incomplete_mapping = load_incomplete_questions(incomplete_questions_file)
    
    results = []
    
    for g_idx, group_data in enumerate(groups):
        group_result = {
            "group_id": g_idx,
            "original_question_id": group_data.get("original_question_id"),
            "dataset": group_data.get("dataset"),
        }
        
        # Compute alignment for each layer
        for layer_name, layer_idx in LAYER_MAP.items():
            out = compute_group_alignment(group_data, incomplete_mapping, layer_idx)
            group_result[f"align_{layer_name}"] = out["alignment"]
            group_result[f"template_shift_norm_{layer_name}"] = out["template_shift_norm"]
            group_result[f"original_shift_norm_{layer_name}"] = out["original_shift_norm"]
        
        results.append(group_result)
    
    return results


#6. Compute summary statistics
def compute_summary_stats(results: List[Dict]) -> Dict:
    """
    Compute summary statistics for alignment scores across groups and by dataset.
    Returns a dictionary with overall and per-dataset statistics.
    """
    summary = {}
    summary_by_dataset = {}
    
    # Get all unique datasets
    datasets = set(r.get("dataset") for r in results if r.get("dataset"))
    
    # Compute statistics for each layer
    for layer_name in LAYER_MAP.keys():
        # Overall statistics (all datasets combined)
        align_values = [r[f"align_{layer_name}"] for r in results 
                       if f"align_{layer_name}" in r and not np.isnan(r[f"align_{layer_name}"])]
        
        if len(align_values) > 0:
            align_array = np.array(align_values)
            # Count high alignment scores (>= 0.8) indicating strong directional collapse
            num_high_align = np.sum(align_array >= 0.8)
            
            summary[layer_name] = {
                "num_groups": len(align_values),
                "mean": float(align_array.mean()),
                "std": float(align_array.std()),
                "min": float(align_array.min()),
                "max": float(align_array.max()),
                "median": float(np.median(align_array)),
                "num_groups_high_align": int(num_high_align),
                "pct_groups_high_align": float((num_high_align / len(align_array)) * 100)
            }
        else:
            summary[layer_name] = {
                "num_groups": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
                "median": None,
                "num_groups_high_align": 0,
                "pct_groups_high_align": 0.0
            }
        
        # Per-dataset statistics
        summary_by_dataset[layer_name] = {}
        for dataset in datasets:
            dataset_align_values = [
                r[f"align_{layer_name}"] for r in results 
                if r.get("dataset") == dataset and f"align_{layer_name}" in r 
                and not np.isnan(r[f"align_{layer_name}"])
            ]
            
            if len(dataset_align_values) > 0:
                dataset_align_array = np.array(dataset_align_values)
                dataset_num_high_align = np.sum(dataset_align_array >= 0.8)
                
                summary_by_dataset[layer_name][dataset] = {
                    "num_groups": len(dataset_align_values),
                    "mean": float(dataset_align_array.mean()),
                    "std": float(dataset_align_array.std()),
                    "min": float(dataset_align_array.min()),
                    "max": float(dataset_align_array.max()),
                    "median": float(np.median(dataset_align_array)),
                    "num_groups_high_align": int(dataset_num_high_align),
                    "pct_groups_high_align": float((dataset_num_high_align / len(dataset_align_array)) * 100)
                }
            else:
                summary_by_dataset[layer_name][dataset] = {
                    "num_groups": 0,
                    "mean": None,
                    "std": None,
                    "min": None,
                    "max": None,
                    "median": None,
                    "num_groups_high_align": 0,
                    "pct_groups_high_align": 0.0
                }
    
    return {
        "overall": summary,
        "by_dataset": summary_by_dataset
    }

#7. Report metrics
def report_results(results: List[Dict], summary_stats: Dict):
    """
    Report distribution of alignment scores across groups.
    Also reports per-dataset statistics.
    """
    print("\n" + "="*80)
    print("Directional Collapse Experiment Results - Overall")
    print("="*80)
    
    # Overall statistics
    for layer_name in LAYER_MAP.keys():
        stats = summary_stats["overall"][layer_name]
        
        if stats["num_groups"] == 0:
            print(f"\nLayer: {layer_name} - No valid alignment values")
            continue
        
        print(f"\nLayer: {layer_name}")
        print(f"  Number of groups: {stats['num_groups']}")
        print(f"  Mean alignment: {stats['mean']:.4f}")
        print(f"  Std alignment: {stats['std']:.4f}")
        print(f"  Min alignment: {stats['min']:.4f}")
        print(f"  Max alignment: {stats['max']:.4f}")
        print(f"  Median alignment: {stats['median']:.4f}")
        print(f"  Groups with high alignment (>= 0.8): {stats['num_groups_high_align']} ({stats['pct_groups_high_align']:.2f}%)")
    
    # Per-dataset statistics
    print("\n" + "="*80)
    print("Directional Collapse Experiment Results - By Dataset")
    print("="*80)
    
    # Get all datasets
    datasets = set()
    for layer_name in LAYER_MAP.keys():
        datasets.update(summary_stats["by_dataset"][layer_name].keys())
    
    for dataset in sorted(datasets):
        print(f"\nDataset: {dataset}")
        for layer_name in LAYER_MAP.keys():
            stats = summary_stats["by_dataset"][layer_name].get(dataset)
            if stats and stats["num_groups"] > 0:
                print(f"  Layer {layer_name}:")
                print(f"    Number of groups: {stats['num_groups']}")
                print(f"    Mean alignment: {stats['mean']:.4f}")
                print(f"    Std alignment: {stats['std']:.4f}")
                print(f"    Min alignment: {stats['min']:.4f}")
                print(f"    Max alignment: {stats['max']:.4f}")
                print(f"    Median alignment: {stats['median']:.4f}")
                print(f"    Groups with high alignment (>= 0.8): {stats['num_groups_high_align']} ({stats['pct_groups_high_align']:.2f}%)")
    
    print("\n" + "="*80)


# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate Directional Collapse (alignment with template shift) metrics"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="talzoomanzoo/contamination-finals-qwen2.5-0.5b-merged",
        help="Model name or path to use for embedding extraction (default: talzoomanzoo/contamination-finals-qwen2.5-0.5b-merged)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for results (default: dc_results.json in script directory)"
    )
    parser.add_argument(
        "--similar-questions",
        type=str,
        default=None,
        help="Path to similar_questions.json file (default: similar_questions.json in script directory)"
    )
    parser.add_argument(
        "--incomplete-questions",
        type=str,
        default=None,
        help="Path to incomplete_questions.jsonl file (default: incomplete_questions.jsonl in script directory)"
    )
    
    args = parser.parse_args()
    
    # File paths relative to script directory
    script_dir = Path(__file__).parent
    similar_questions_file = args.similar_questions or str(script_dir / "similar_questions.json")
    incomplete_questions_file = args.incomplete_questions or str(script_dir / "incomplete_questions.jsonl")
    output_file = args.output or str(script_dir / "dc_results.json")
    
    # Initialize model
    print(f"Loading model: {args.model_name}")
    initialize_model(args.model_name)
    print(f"Model loaded. Layer map: {LAYER_MAP}")
    
    # Run experiment
    results = run_alignment_experiment(similar_questions_file, incomplete_questions_file)
    
    # Compute summary statistics
    summary_stats = compute_summary_stats(results)
    
    # Report results
    report_results(results, summary_stats)
    
    # Save results to file (including summary statistics)
    output_data = {
        "model_name": args.model_name,
        "groups": results,
        "summary": summary_stats
    }
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")

