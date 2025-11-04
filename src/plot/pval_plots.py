#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy import stats
from pathlib import Path
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split


def load_experiment_config(results_dir):
    """Load experiment configuration from experiment_config.json"""
    config_path = Path(results_dir) / "experiment_config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def load_mean_directional_derivatives(results_dir):
    """Load and compute mean directional derivatives across all seeds"""
    list_of_mean_weighted_stabilities = []
    
    for subdir in os.listdir(results_dir):
        if 'seed' in subdir:
            print(f"Loading: {Path(results_dir) / subdir}")
            weighted_sens = np.load(Path(results_dir) / subdir / "weighted_sens.npy")
            list_of_mean_weighted_stabilities.append(weighted_sens.mean(axis=0))
    
    return np.vstack(list_of_mean_weighted_stabilities)


def compute_statistics(mean_directional_derivatives):
    """Compute t-statistics and p-values"""
    t_stats, p_values = stats.ttest_1samp(
        mean_directional_derivatives,
        0,  # null hypothesis: mean = 0
        axis=0,
        alternative='two-sided'
    )
    return t_stats, p_values


def get_significant_concepts(mean_per_concept, p_values, concept_texts, n_concepts=20, alpha=0.05):
    """Get top N significant concepts in positive and negative directions"""
    n_total_concepts = len(concept_texts)
    bonferroni_threshold = alpha / n_total_concepts
    
    # Get indices of significant concepts
    significant_mask = p_values < bonferroni_threshold
    
    # Sort by mean directional derivative
    sorted_indices = np.argsort(mean_per_concept)
    
    # Get top positive (highest mean)
    positive_concepts = []
    for idx in sorted_indices[::-1]:
        if significant_mask[idx]:
            positive_concepts.append({
                'rank': len(positive_concepts) + 1,
                'concept': concept_texts[idx],
                'mean_dd': mean_per_concept[idx],
                'p_value': p_values[idx],
                'concept_idx': idx
            })
        if len(positive_concepts) >= n_concepts:
            break
    
    # Get top negative (lowest mean)
    negative_concepts = []
    for idx in sorted_indices:
        if significant_mask[idx]:
            negative_concepts.append({
                'rank': len(negative_concepts) + 1,
                'concept': concept_texts[idx],
                'mean_dd': mean_per_concept[idx],
                'p_value': p_values[idx],
                'concept_idx': idx
            })
        if len(negative_concepts) >= n_concepts:
            break
    
    return positive_concepts, negative_concepts


def save_concept_tables(positive_concepts, negative_concepts, results_dir):
    """Save concept tables to CSV files"""
    output_dir = Path(results_dir) / "analysis_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save positive concepts
    df_pos = pd.DataFrame(positive_concepts)
    df_pos = df_pos[['rank', 'concept', 'mean_dd', 'p_value']]
    pos_path = output_dir / "top_20_positive_concepts.csv"
    df_pos.to_csv(pos_path, index=False)
    print(f"Saved: {pos_path}")
    
    # Save negative concepts
    df_neg = pd.DataFrame(negative_concepts)
    df_neg = df_neg[['rank', 'concept', 'mean_dd', 'p_value']]
    neg_path = output_dir / "top_20_negative_concepts.csv"
    df_neg.to_csv(neg_path, index=False)
    print(f"Saved: {neg_path}")
    
    return output_dir


def center_crop_to_square(img):
    """Center crop image to a square using the smaller dimension"""
    width, height = img.size
    size = min(width, height)
    left = (width - size) // 2
    top = (height - size) // 2
    right = left + size
    bottom = top + size
    return img.crop((left, top, right, bottom))


def create_concept_visualization(concept_idx, concept_name, top_indices, bottom_indices,
                               concept_activations, probe_paths, n_images, save_path):
    """Create and save visualization for a single concept"""
    fig, axes = plt.subplots(2, n_images, figsize=(16, 10))
    fig.suptitle(f'Concept: {concept_name}', fontsize=16, fontweight='bold', y=0.75)
    
    # Ensure axes is 2D
    if n_images == 1:
        axes = axes.reshape(2, 1)
    
    # Top activated images
    for i, img_idx in enumerate(top_indices):
        try:
            img_path = probe_paths[img_idx]
            img = Image.open(img_path).convert('RGB')
            img_cropped = center_crop_to_square(img)
            axes[0, i].imshow(img_cropped)
            axes[0, i].axis('off')
        except Exception as e:
            axes[0, i].text(0.5, 0.5, f'Error loading\n{Path(probe_paths[img_idx]).name}', 
                          ha='center', va='center', transform=axes[0, i].transAxes)
            axes[0, i].axis('off')
    
    # Bottom activated images  
    for i, img_idx in enumerate(bottom_indices):
        try:
            img_path = probe_paths[img_idx]
            img = Image.open(img_path).convert('RGB')
            img_cropped = center_crop_to_square(img)
            axes[1, i].imshow(img_cropped)
            axes[1, i].axis('off')
        except Exception as e:
            axes[1, i].text(0.5, 0.5, f'Error loading\n{Path(probe_paths[img_idx]).name}', 
                          ha='center', va='center', transform=axes[1, i].transAxes)
            axes[1, i].axis('off')
    
    # Add row labels
    fig.text(0.06, 0.60, 'Most Activating\nImages', ha='center', va='center', 
             fontsize=12, fontweight='bold', rotation=90)
    fig.text(0.06, 0.35, 'Least Activating\nImages', ha='center', va='center', 
             fontsize=12, fontweight='bold', rotation=90)
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.12, top=0.9, bottom=0.05, hspace=-0.6)
    
    # Save with 300 DPI
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization: {save_path}")
    plt.close()


def prepare_image_paths(config, seed=0):
    """
    Prepare train and test image paths based on config.
    Must match the DDIDataLoader logic from the experiment script.
    
    Args:
        config: Experiment configuration
        seed: Random seed to use (default 0 for seed_0 folder similarity matrix)
    """
    metadata_path = config['metadata_path']
    base_dir = config['data_base_dir']
    test_size = config['test_size']
    demo_size = config['demo_size']
    use_demos = config['prompt_config'].get('use_demos', False)
    
    # Load metadata
    df = pd.read_csv(metadata_path, index_col=0)
    
    # This matches DDIDataLoader.__init__ logic
    # First split: train/test
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed  # Use the seed parameter
    )
    
    # Second split: demos (if needed)
    if use_demos:
        train_df, demo_df = train_test_split(
            train_df,
            test_size=demo_size,
            random_state=seed  # Use the same seed
        )
        print(f"Using ICL with {len(demo_df)} demo images")
    
    # Create probe paths (train set after demo split)
    probe_paths = [Path(base_dir) / file for file in train_df.DDI_file]
    
    return probe_paths


def generate_visualizations_for_concepts(concepts, sim_matrix, probe_paths, 
                                        concept_texts, output_dir, n_images=7):
    """Generate visualizations for a list of concepts"""
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    for concept_data in concepts:
        concept_idx = concept_data['concept_idx']
        concept_name = concept_data['concept']
        
        # Get activation scores for this concept
        concept_activations = sim_matrix[concept_idx, :]
        
        # Get indices sorted by activation
        sorted_indices = np.argsort(-concept_activations)
        top_indices = sorted_indices[:n_images]
        bottom_indices = sorted_indices[-n_images:]
        
        # Create safe filename
        safe_name = "".join(c for c in concept_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_name = safe_name[:50]  # Limit length
        save_path = viz_dir / f"concept_{concept_data['rank']:02d}_{safe_name}.png"
        
        create_concept_visualization(
            concept_idx, concept_name, top_indices, bottom_indices,
            concept_activations, probe_paths, n_images, save_path
        )


def main(results_dir, n_concepts=20, n_images=7):
    """Main analysis pipeline"""
    results_dir = Path(results_dir)
    
    print("="*60)
    print(f"Starting analysis for: {results_dir}")
    print("="*60)
    
    # Load experiment config
    print("\n1. Loading experiment configuration...")
    exp_config = load_experiment_config(results_dir)
    
    # Load directional derivatives
    print("\n2. Loading mean directional derivatives...")
    mean_directional_derivatives = load_mean_directional_derivatives(results_dir)
    print(f"   Shape: {mean_directional_derivatives.shape}")
    
    # Compute statistics
    print("\n3. Computing statistics...")
    t_stats, p_values = compute_statistics(mean_directional_derivatives)
    mean_per_concept = np.mean(mean_directional_derivatives, axis=0)
    
    n_total = len(p_values)
    n_significant = ((p_values * n_total) < 0.05).sum()
    print(f"   Total concepts: {n_total}")
    print(f"   Significant concepts (Bonferroni corrected): {n_significant}")
    
    # Load concept texts
    print("\n4. Loading concept texts...")
    with open(results_dir / "seed_0" / "concept_texts.json", "r") as f:
        concept_texts = json.load(f)
    
    # Get significant concepts
    print(f"\n5. Extracting top {n_concepts} significant concepts...")
    positive_concepts, negative_concepts = get_significant_concepts(
        mean_per_concept, p_values, concept_texts, n_concepts=n_concepts
    )
    print(f"   Found {len(positive_concepts)} positive and {len(negative_concepts)} negative concepts")
    
    # Save tables
    print("\n6. Saving concept tables to CSV...")
    output_dir = save_concept_tables(positive_concepts, negative_concepts, results_dir)
    
    # Prepare image paths
    print("\n7. Preparing image paths...")
    # Use seed 0 to match the seed_0 folder from which we load similarity_matrix
    probe_paths = prepare_image_paths(exp_config, seed=0)
    print(f"   Loaded {len(probe_paths)} probe images")
    
    # Load similarity matrix
    print("\n8. Loading similarity matrix...")
    sim_matrix = np.load(results_dir / "seed_0" / "similarity_matrix.npy")
    print(f"   Shape: {sim_matrix.shape}")
    
    # Generate visualizations
    print(f"\n9. Generating visualizations ({n_images} images per concept)...")
    print("   Positive concepts:")
    generate_visualizations_for_concepts(
        positive_concepts, sim_matrix, probe_paths, concept_texts, output_dir, n_images
    )
    print("   Negative concepts:")
    generate_visualizations_for_concepts(
        negative_concepts, sim_matrix, probe_paths, concept_texts, output_dir, n_images
    )
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    # Set your experiment directory here
    results_dirs = '/home/groups/roxanad/sonnet/vcr/results'
    
    # Run analysis on all results in results_dir
    for results_dir in os.listdir(results_dirs):
        full_path = os.path.join(results_dirs, results_dir)
        if os.path.isdir(full_path) and not os.pathexists(full_path, 'analysis_outputs'):
            main(full_path, n_concepts=20, n_images=7)