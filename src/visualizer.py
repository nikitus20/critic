"""Simple visualization for DeltaBench results."""

import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional


def plot_results(results_df: pd.DataFrame, save_path: Optional[str] = None):
    """Plot basic evaluation results."""
    if len(results_df) == 0:
        print("No results to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('DeltaBench Results', fontsize=14)
    
    # F1 Score Distribution
    axes[0, 0].hist(results_df['f1_score'], bins=15, alpha=0.7, color='skyblue')
    axes[0, 0].set_title('F1 Score Distribution')
    axes[0, 0].set_xlabel('F1 Score')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].axvline(results_df['f1_score'].mean(), color='red', linestyle='--')
    
    # Precision vs Recall
    axes[0, 1].scatter(results_df['recall'], results_df['precision'], alpha=0.6)
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision vs Recall')
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].set_ylim(0, 1)
    
    # Error Counts
    axes[1, 0].bar(['Predicted', 'True'], 
                   [results_df['predicted_errors'].sum(), results_df['true_errors'].sum()],
                   color=['lightcoral', 'lightgreen'])
    axes[1, 0].set_title('Total Errors')
    axes[1, 0].set_ylabel('Count')
    
    # Token Usage
    axes[1, 1].hist(results_df['tokens_used'], bins=15, alpha=0.7, color='gold')
    axes[1, 1].set_title('Token Usage')
    axes[1, 1].set_xlabel('Tokens')
    axes[1, 1].set_ylabel('Count')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()