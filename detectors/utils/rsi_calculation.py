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

def load_incomplete_variants(file_path: str) -> List[Dict]:
    """
    Load incomplete_variants_rsi.json file.
    Returns list of entries with incomplete variants.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def organize_variants_by_group(variants_data: List[Dict]) -> Dict[int, Dict]:
    """
    Organize incomplete variants by original_question_id.
    
    Returns:
        Dict mapping original_question_id -> {
            'original': dict with variants for original question,
            'similars': dict mapping similar_id -> dict with variants
        }
    """
    groups = {}
    
    for entry in variants_data:
        orig_id = entry['original_question_id']
        question_type = entry['type']
        
        if orig_id not in groups:
            groups[orig_id] = {
                'original': None,
                'similars': {}
            }
        
        # Extract variant texts (just the variant strings)
        variant_texts = [v['variant'] for v in entry['variants']]
        
        entry_data = {
            'incomplete_question': entry['original_incomplete_question'],
            'variant_texts': variant_texts,
            'id': entry['id']
        }
        
        if question_type == 'original':
            groups[orig_id]['original'] = entry_data
        else:  # similar
            similar_id = entry['id']
            groups[orig_id]['similars'][similar_id] = entry_data
    
    return groups

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

#extract 
@torch.no_grad()
def extract_variant_reps(
    texts: list[str],
    layer_idx: int,
) -> torch.Tensor:
    """
    Returns:
        reps: (M, D) tensor
    """
    reps = []
    for t in texts:
        reps.append(extract_representation(t, layer_idx))
    return torch.stack(reps, dim=0)


#5. Metric3: Representation Stiffness Index(RSI) Calculation

#compute RSI for one question, one layer
def compute_rsi_single(
    incomplete_variants: list[str],
    layer_idx: int,
) -> float:
    """
    RSI_{i,l}^{(g)}
    """
    reps = extract_variant_reps(incomplete_variants, layer_idx)  # (M, D)

    mean_rep = reps.mean(dim=0, keepdim=True)  # (1, D)

    # distances to mean
    dists = torch.norm(reps - mean_rep, dim=1)  # (M,)

    return dists.mean().item()


#compute zRSI for one group
def compute_group_zrsi(
    group_data: dict,
    layer_idx: int,
):
    """
    Compute RSI and zRSI for one group at one layer.
    
    Args:
        group_data: dict with 'original' and 'similars' keys
        layer_idx: which layer to extract representations from
        
    Returns:
        dict with RSI values and zRSI score
    """
    # Compute RSI for original
    original_variants = group_data['original']['variant_texts']
    rsi_original = compute_rsi_single(original_variants, layer_idx)
    
    # Compute RSI for each similar
    rsi_similars = []
    similar_ids = []
    for sim_id, sim_data in group_data['similars'].items():
        sim_variants = sim_data['variant_texts']
        rsi_sim = compute_rsi_single(sim_variants, layer_idx)
        rsi_similars.append(rsi_sim)
        similar_ids.append(sim_id)
    
    rsi_similars = np.array(rsi_similars)
    
    # Compute z-score
    mu = rsi_similars.mean()
    sigma = rsi_similars.std(ddof=1) + 1e-8
    z_rsi = (rsi_original - mu) / sigma
    
    return {
        "rsi_original": float(rsi_original),
        "rsi_similars": rsi_similars.tolist(),
        "similar_ids": similar_ids,
        "mean_similars": float(mu),
        "std_similars": float(sigma),
        "zRSI": float(z_rsi),
    }


def compute_all_groups(
    groups: Dict[int, Dict],
    layer_map: Dict[str, int]
) -> Dict:
    """
    Compute RSI metrics for all groups across all layers.
    
    Args:
        groups: organized variant data by group
        layer_map: dict mapping layer names to indices
        
    Returns:
        dict with results per group and per layer
    """
    results = {}
    
    for orig_id, group_data in groups.items():
        if group_data['original'] is None or len(group_data['similars']) == 0:
            print(f"Skipping group {orig_id}: missing original or similars")
            continue
            
        print(f"Processing group {orig_id}...")
        results[orig_id] = {}
        
        for layer_name, layer_idx in layer_map.items():
            print(f"  Layer {layer_name} (idx={layer_idx})...")
            layer_results = compute_group_zrsi(group_data, layer_idx)
            results[orig_id][layer_name] = layer_results
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compute Representation Stiffness Index (RSI) metrics"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Hugging Face model name (e.g., 'meta-llama/Llama-2-7b-hf')"
    )
    parser.add_argument(
        "--similar_questions",
        type=str,
        default="similar_questions.json",
        help="Path to similar_questions.json"
    )
    parser.add_argument(
        "--incomplete_variants",
        type=str,
        default="incomplete_variants_rsi.json",
        help="Path to incomplete_variants_rsi.json"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="rsi_results.json",
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    # Initialize model
    print(f"Loading model: {args.model_name}")
    initialize_model(args.model_name)
    print(f"Using layers: {LAYER_MAP}")
    
    # Load data
    print(f"\nLoading data...")
    similar_questions = load_similar_questions(args.similar_questions)
    print(f"Loaded {len(similar_questions)} question groups")
    
    variants_data = load_incomplete_variants(args.incomplete_variants)
    print(f"Loaded {len(variants_data)} variant entries")
    
    # Organize by group
    groups = organize_variants_by_group(variants_data)
    print(f"Organized into {len(groups)} groups")
    
    # Compute RSI metrics
    print(f"\nComputing RSI metrics...")
    results = compute_all_groups(groups, LAYER_MAP)
    
    # Save results
    output_data = {
        "model_name": args.model_name,
        "layer_map": LAYER_MAP,
        "results": results
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    
    # Print summary statistics
    print("\n=== Summary ===")
    for layer_name in LAYER_MAP.keys():
        zrsi_values = [
            results[gid][layer_name]['zRSI']
            for gid in results.keys()
        ]
        print(f"{layer_name}: mean zRSI = {np.mean(zrsi_values):.3f}, "
              f"std = {np.std(zrsi_values):.3f}")


if __name__ == "__main__":
    main()
