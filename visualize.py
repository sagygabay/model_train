#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization Script for Deep Learning Model Comparison

Generates publication-ready plots from raw training results saved by
train_evaluate.py, applying styles defined in a YAML configuration file.
"""

import argparse
import os
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import yaml
import logging
from sklearn.metrics import confusion_matrix

# Setup logging
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)

# --- Configuration Loading ---

DEFAULT_STYLE_CONFIG = {
    'Font': {'size': 12, 'family': 'sans-serif', 'title_size': 14, 'label_size': 12, 'tick_size': 10, 'legend_size': 10},
    'Colors': {'palette': 'tab10', 'custom_colors': {}},
    'Lines': {'train_style': '-', 'val_style': '--', 'line_width': 1.5, 'error_band_alpha': 0.2},
    'Plot': {'figsize': [8, 5], 'dpi': 300, 'grid': True, 'grid_style': '--', 'grid_alpha': 0.6, 'legend_loc': 'best'},
    'ConfusionMatrix': {'cmap': 'Blues', 'annot_fmt': 'd', 'annot_kws': {'size': 10}},
    'BarChart': {'width': 0.8},
    'Output': {'vector_format': 'pdf', 'raster_format': 'png', 'raster_dpi': 300, 'transparent_background': False}
}

def load_style_config(config_path):
    """Loads style configuration from YAML file, using defaults for missing keys."""
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
            # Deep merge with defaults (user config takes precedence)
            style_config = merge_dicts(DEFAULT_STYLE_CONFIG.copy(), user_config)
            logger.info(f"Loaded style configuration from: {config_path}")
            return style_config
        except Exception as e:
            logger.error(f"Error loading style config {config_path}: {e}. Using default styles.")
            return DEFAULT_STYLE_CONFIG
    else:
        logger.info("Style config file not found or not specified. Using default styles.")
        return DEFAULT_STYLE_CONFIG

def merge_dicts(base, update):
    """Recursively merge dictionaries. 'update' values override 'base' values."""
    if not isinstance(update, dict): # Handle case where update is not a dict (e.g., null in YAML)
        return base
    for key, value in update.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            base[key] = merge_dicts(base[key], value)
        else:
            base[key] = value
    return base

def apply_plot_style(style_config):
    """Applies the loaded style configuration to matplotlib."""
    font_cfg = style_config.get('Font', {})
    plot_cfg = style_config.get('Plot', {})

    plt.rcParams.update({
        'font.size': font_cfg.get('size', 12),
        'font.family': font_cfg.get('family', 'sans-serif'),
        'axes.titlesize': font_cfg.get('title_size', 14),
        'axes.labelsize': font_cfg.get('label_size', 12),
        'xtick.labelsize': font_cfg.get('tick_size', 10),
        'ytick.labelsize': font_cfg.get('tick_size', 10),
        'legend.fontsize': font_cfg.get('legend_size', 10),
        'figure.dpi': plot_cfg.get('dpi', 300),
        'savefig.dpi': plot_cfg.get('dpi', 300),
        'savefig.transparent': style_config.get('Output', {}).get('transparent_background', False),
        'axes.grid': plot_cfg.get('grid', True),
        'grid.linestyle': plot_cfg.get('grid_style', '--'),
        'grid.alpha': plot_cfg.get('grid_alpha', 0.6),
    })
    # Add specific font family if needed (e.g., Times New Roman)
    # Note: The font must be installed on the system
    specific_family = font_cfg.get('family_specific', None)
    if specific_family:
         plt.rcParams['font.family'] = specific_family


# --- Data Loading ---

def load_model_data(model_dir):
    """Loads history, summary, and predictions for a single model."""
    model_name = model_dir.name
    data = {'name': model_name, 'path': model_dir}
    results_dir = model_dir / "results"

    # Load training history
    history_path = results_dir / "training_history.csv"
    if history_path.exists():
        try:
            data['history'] = pd.read_csv(history_path)
        except Exception as e:
            logger.warning(f"Could not load history for {model_name}: {e}")
            data['history'] = None
    else:
        logger.warning(f"History file not found for {model_name}: {history_path}")
        data['history'] = None

    # Load model summary
    summary_path = model_dir / "model_summary.json"
    if summary_path.exists():
        try:
            with open(summary_path, 'r') as f:
                data['summary'] = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load summary for {model_name}: {e}")
            data['summary'] = {}
    else:
        logger.warning(f"Summary file not found for {model_name}: {summary_path}")
        data['summary'] = {}

    # Load validation predictions
    preds_path = results_dir / "best_model_validation_preds.npz"
    if preds_path.exists():
        try:
            preds_data = np.load(preds_path)
            data['true_labels'] = preds_data['true_labels']
            data['pred_labels'] = preds_data['pred_labels']
        except Exception as e:
            logger.warning(f"Could not load predictions for {model_name}: {e}")
            data['true_labels'], data['pred_labels'] = None, None
    else:
        logger.warning(f"Predictions file not found for {model_name}: {preds_path}")
        data['true_labels'], data['pred_labels'] = None, None

    return data

def find_model_dirs(results_base_dir):
    """Finds valid model result directories."""
    model_dirs = []
    base_path = Path(results_base_dir)
    if not base_path.is_dir():
        logger.error(f"Results directory not found: {results_base_dir}")
        return []

    for item in base_path.iterdir():
        # Check if it's a directory and contains expected files/subdirs
        if item.is_dir():
            results_subdir = item / "results"
            if results_subdir.exists() and (results_subdir / "training_history.csv").exists():
                 model_dirs.append(item)
            else:
                 # Also check if it's the visualizations dir itself
                 if item.name != 'visualizations':
                      logger.debug(f"Skipping directory (missing results): {item}")

    if not model_dirs:
         logger.warning(f"No valid model result directories found in {results_base_dir}")

    return model_dirs


# --- Plotting Functions ---

def get_colors(model_names, style_config):
    """Gets colors for models, using palette and custom overrides."""
    color_cfg = style_config.get('Colors', {})
    palette_name = color_cfg.get('palette', 'tab10')
    # Ensure custom_colors is a dictionary, default to empty if None or missing
    custom_colors = color_cfg.get('custom_colors') or {} # <-- FIX APPLIED HERE

    try:
        # Attempt to use seaborn palettes first for more options
        base_colors = sns.color_palette(palette_name, n_colors=len(model_names)).as_hex()
    except ValueError:
        # Fallback to matplotlib palettes
        try:
            cmap = plt.get_cmap(palette_name)
            base_colors = [plt.cm.colors.to_hex(cmap(i)) for i in np.linspace(0, 1, len(model_names))]
        except ValueError:
            logger.warning(f"Invalid palette name: {palette_name}. Falling back to 'tab10'.")
            base_colors = sns.color_palette('tab10', n_colors=len(model_names)).as_hex()

    # Apply custom overrides
    final_colors = {}
    for i, name in enumerate(model_names):
        final_colors[name] = custom_colors.get(name, base_colors[i % len(base_colors)]) # Cycle through base colors if needed

    return final_colors

def save_plot(fig, output_dir, filename_base, style_config):
    """Saves the plot in specified vector and raster formats."""
    output_cfg = style_config.get('Output', {})
    vec_fmt = output_cfg.get('vector_format', 'pdf')
    ras_fmt = output_cfg.get('raster_format', 'png')
    ras_dpi = output_cfg.get('raster_dpi', 300)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save vector format
    vec_path = output_dir / f"{filename_base}.{vec_fmt}"
    try:
        fig.savefig(vec_path, bbox_inches='tight')
        logger.info(f"Saved {vec_fmt.upper()} plot: {vec_path}")
    except Exception as e:
        logger.error(f"Error saving {vec_fmt.upper()} plot {vec_path}: {e}")

    # Save raster format
    ras_path = output_dir / f"{filename_base}.{ras_fmt}"
    try:
        fig.savefig(ras_path, dpi=ras_dpi, bbox_inches='tight')
        logger.info(f"Saved {ras_fmt.upper()} plot: {ras_path}")
    except Exception as e:
        logger.error(f"Error saving {ras_fmt.upper()} plot {ras_path}: {e}")


def plot_train_val_curves(all_models_data, style_config, output_dir):
    """Plots combined training/validation accuracy and loss curves."""
    logger.info("Generating Training/Validation Accuracy and Loss plots...")
    line_cfg = style_config.get('Lines', {})
    plot_cfg = style_config.get('Plot', {})
    model_names = [m['name'] for m in all_models_data]
    colors = get_colors(model_names, style_config)

    metrics_to_plot = [('acc', 'Accuracy (%)', 'Accuracy'), ('loss', 'Loss', 'Loss')]

    for metric_key, y_label, title_suffix in metrics_to_plot:
        fig, ax = plt.subplots(figsize=plot_cfg.get('figsize', [8, 5]))
        apply_plot_style(style_config) # Apply style for each new figure

        for model_data in all_models_data:
            name = model_data['name']
            history = model_data['history']
            if history is None or history.empty:
                logger.warning(f"Skipping {name} in {title_suffix} plot (no history data).")
                continue

            color = colors.get(name, 'gray')
            epochs = history['epoch']
            train_metric = history[f'train_{metric_key}']
            val_metric = history[f'val_{metric_key}']

            # Plot train curve
            ax.plot(epochs, train_metric * 100 if metric_key == 'acc' else train_metric,
                    label=f'{name} Train', color=color,
                    linestyle=line_cfg.get('train_style', '-'),
                    linewidth=line_cfg.get('line_width', 1.5))

            # Plot validation curve
            ax.plot(epochs, val_metric * 100 if metric_key == 'acc' else val_metric,
                    label=f'{name} Val', color=color,
                    linestyle=line_cfg.get('val_style', '--'),
                    linewidth=line_cfg.get('line_width', 1.5))

        ax.set_xlabel('Epochs')
        ax.set_ylabel(y_label)
        ax.set_title(f'Training & Validation {title_suffix}')
        ax.legend(loc=plot_cfg.get('legend_loc', 'best'))
        ax.grid(plot_cfg.get('grid', True), linestyle=plot_cfg.get('grid_style', '--'), alpha=plot_cfg.get('grid_alpha', 0.6))
        if metric_key == 'acc':
             ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100.0))
             ax.set_ylim(bottom=max(0, ax.get_ylim()[0])) # Ensure y-axis starts at 0 or above

        plt.tight_layout()
        save_plot(fig, output_dir, f'combined_train_val_{metric_key}', style_config)
        plt.close(fig)


def plot_convergence_speed(all_models_data, style_config, output_dir):
    """Plots bar chart of epochs to reach 95% of peak validation accuracy."""
    logger.info("Generating Convergence Speed plot...")
    plot_cfg = style_config.get('Plot', {})
    bar_cfg = style_config.get('BarChart', {})
    model_names = [m['name'] for m in all_models_data]
    colors = get_colors(model_names, style_config)
    convergence_threshold_factor = 0.95 # Default 95%

    convergence_epochs = {}
    for model_data in all_models_data:
        name = model_data['name']
        history = model_data['history']
        if history is None or history.empty or 'val_acc' not in history.columns:
            convergence_epochs[name] = np.nan # Mark as N/A
            continue

        peak_val_acc = history['val_acc'].max()
        target_acc = peak_val_acc * convergence_threshold_factor

        # Find the first epoch where validation accuracy reaches the target
        converged_epoch_series = history[history['val_acc'] >= target_acc]['epoch']
        if not converged_epoch_series.empty:
            convergence_epochs[name] = converged_epoch_series.iloc[0]
        else:
            convergence_epochs[name] = np.nan # Did not converge

    # Prepare data for plotting
    plot_data = pd.DataFrame.from_dict(convergence_epochs, orient='index', columns=['epochs_to_converge'])
    plot_data = plot_data.reindex(model_names) # Keep original order
    plot_data['color'] = [colors.get(name, 'gray') for name in plot_data.index]

    fig, ax = plt.subplots(figsize=plot_cfg.get('figsize', [8, 5]))
    apply_plot_style(style_config)

    bars = ax.bar(plot_data.index, plot_data['epochs_to_converge'],
                  color=plot_data['color'], width=bar_cfg.get('width', 0.8))

    ax.set_xlabel('Model')
    ax.set_ylabel(f'Epochs to Reach {convergence_threshold_factor*100:.0f}% of Peak Val Acc')
    ax.set_title('Convergence Speed Comparison')
    plt.xticks(rotation=45, ha='right')
    ax.grid(plot_cfg.get('grid', True), linestyle=plot_cfg.get('grid_style', '--'), alpha=plot_cfg.get('grid_alpha', 0.6), axis='y')

    # Add labels to bars (handle NaN)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if pd.notna(height):
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=style_config.get('Font', {}).get('tick_size', 10))
        else:
             # Add N/A text slightly above 0
             ax.text(bar.get_x() + bar.get_width() / 2., 0.1, 'N/A', ha='center', va='bottom', fontsize=style_config.get('Font', {}).get('tick_size', 10), color='red')

    # Adjust y-limit if needed
    if plot_data['epochs_to_converge'].notna().any():
         ax.set_ylim(bottom=0, top=plot_data['epochs_to_converge'].max() * 1.1)
    else:
         ax.set_ylim(bottom=0, top=1) # Set a minimal top limit if all are NaN


    plt.tight_layout()
    save_plot(fig, output_dir, 'comparison_convergence_speed', style_config)
    plt.close(fig)


def plot_confusion_matrix(model_data, style_config, output_dir):
    """Plots the confusion matrix for a single model."""
    name = model_data['name']
    true_labels = model_data.get('true_labels')
    pred_labels = model_data.get('pred_labels')
    cm_cfg = style_config.get('ConfusionMatrix', {})
    output_cfg = style_config.get('Output', {})

    if true_labels is None or pred_labels is None:
        logger.warning(f"Skipping confusion matrix for {name} (missing prediction data).")
        return

    logger.info(f"Generating Confusion Matrix for {name}...")
    fig, ax = plt.subplots(figsize=style_config.get('Plot', {}).get('figsize', [6, 5])) # Often smaller
    apply_plot_style(style_config)

    try:
        # Determine unique labels from true labels and sort them for consistent order
        unique_labels = sorted(np.unique(true_labels))
        # Ensure labels used for matrix calculation and plotting are consistent
        cm = confusion_matrix(true_labels, pred_labels, labels=unique_labels)
        # Use the determined labels for heatmap axes
        class_names = [str(label) for label in unique_labels] # Convert labels to strings for display

        sns.heatmap(cm, annot=True, fmt=cm_cfg.get('annot_fmt', 'd'), cmap=cm_cfg.get('cmap', 'Blues'),
                    xticklabels=class_names, yticklabels=class_names,
                    annot_kws=cm_cfg.get('annot_kws', {'size': 10}), ax=ax)
        ax.set_title(f'Confusion Matrix - {name}')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')

        plt.tight_layout()
        # Save confusion matrix specifically in raster format if desired
        ras_fmt = output_cfg.get('raster_format', 'png')
        ras_dpi = output_cfg.get('raster_dpi', 300)
        ras_path = output_dir / f"{name}_confusion_matrix.{ras_fmt}"
        try:
            fig.savefig(ras_path, dpi=ras_dpi, bbox_inches='tight')
            logger.info(f"Saved {ras_fmt.upper()} plot: {ras_path}")
        except Exception as e:
            logger.error(f"Error saving {ras_fmt.upper()} plot {ras_path}: {e}")

        plt.close(fig)

    except Exception as e:
        logger.error(f"Error generating confusion matrix for {name}: {e}")
        plt.close(fig)


def plot_lr_schedule(all_models_data, style_config, output_dir):
    """Plots the learning rate schedule used during training."""
    logger.info("Generating Learning Rate Schedule plot...")
    line_cfg = style_config.get('Lines', {})
    plot_cfg = style_config.get('Plot', {})
    model_names = [m['name'] for m in all_models_data]
    colors = get_colors(model_names, style_config)

    fig, ax = plt.subplots(figsize=plot_cfg.get('figsize', [8, 5]))
    apply_plot_style(style_config)
    found_lr_data = False

    for model_data in all_models_data:
        name = model_data['name']
        history = model_data['history']
        if history is None or history.empty or 'lr' not in history.columns:
            logger.warning(f"Skipping {name} in LR plot (no LR data in history).")
            continue

        # Check if LR actually changed
        if history['lr'].nunique() <= 1:
             logger.info(f"Skipping {name} in LR plot (LR did not change).")
             continue

        found_lr_data = True
        color = colors.get(name, 'gray')
        epochs = history['epoch']
        learning_rates = history['lr']

        ax.plot(epochs, learning_rates, label=name, color=color,
                linestyle=line_cfg.get('train_style', '-'), # Use train style for LR
                linewidth=line_cfg.get('line_width', 1.5))

    if not found_lr_data:
         logger.info("No models with changing LR found. Skipping LR plot generation.")
         plt.close(fig)
         return

    ax.set_xlabel('Epochs')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.legend(loc=plot_cfg.get('legend_loc', 'best'))
    ax.grid(plot_cfg.get('grid', True), linestyle=plot_cfg.get('grid_style', '--'), alpha=plot_cfg.get('grid_alpha', 0.6))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1e')) # Scientific notation

    plt.tight_layout()
    save_plot(fig, output_dir, 'combined_lr_schedule', style_config)
    plt.close(fig)


def plot_model_complexity(all_models_data, style_config, output_dir):
    """Plots bar chart of total trainable parameters."""
    logger.info("Generating Model Complexity plot...")
    plot_cfg = style_config.get('Plot', {})
    bar_cfg = style_config.get('BarChart', {})
    model_names = [m['name'] for m in all_models_data]
    colors = get_colors(model_names, style_config)

    complexity_data = {}
    for model_data in all_models_data:
        name = model_data['name']
        summary = model_data.get('summary', {})
        # Use trainable_params if available, fall back to total_params
        params = summary.get('trainable_params', summary.get('total_params', None))
        if params is not None:
            complexity_data[name] = params
        else:
            complexity_data[name] = np.nan

    # Prepare data for plotting
    plot_data = pd.DataFrame.from_dict(complexity_data, orient='index', columns=['parameters'])
    plot_data = plot_data.reindex(model_names) # Keep original order
    plot_data['color'] = [colors.get(name, 'gray') for name in plot_data.index]

    # Convert parameters to millions for better readability
    plot_data['parameters_M'] = plot_data['parameters'] / 1_000_000

    fig, ax = plt.subplots(figsize=plot_cfg.get('figsize', [8, 5]))
    apply_plot_style(style_config)

    bars = ax.bar(plot_data.index, plot_data['parameters_M'],
                  color=plot_data['color'], width=bar_cfg.get('width', 0.8))

    ax.set_xlabel('Model')
    ax.set_ylabel('Trainable Parameters (Millions)')
    ax.set_title('Model Complexity Comparison')
    plt.xticks(rotation=45, ha='right')
    ax.grid(plot_cfg.get('grid', True), linestyle=plot_cfg.get('grid_style', '--'), alpha=plot_cfg.get('grid_alpha', 0.6), axis='y')

    # Add labels to bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if pd.notna(height):
            # Show M for millions
            label_text = f"{height:.1f}M" if height >= 0.1 else f"{plot_data['parameters'].iloc[i]:,}"
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    label_text, ha='center', va='bottom', fontsize=style_config.get('Font', {}).get('tick_size', 10))
        else:
             ax.text(bar.get_x() + bar.get_width() / 2., 0.1, 'N/A', ha='center', va='bottom', fontsize=style_config.get('Font', {}).get('tick_size', 10), color='red')

    if plot_data['parameters_M'].notna().any():
         ax.set_ylim(bottom=0, top=plot_data['parameters_M'].max() * 1.1)
    else:
         ax.set_ylim(bottom=0, top=1)

    # fig.tight_layout() # Try adjusting subplot parameters instead
    try:
        fig.subplots_adjust(bottom=0.25) # Increase bottom margin for rotated labels
    except Exception as e:
        logger.warning(f"Could not apply subplots_adjust: {e}") # Log if adjustment fails
    save_plot(fig, output_dir, 'comparison_model_complexity', style_config)
    plt.close(fig)


def plot_training_time(all_models_data, style_config, output_dir):
    """Plots bar chart of average training time per epoch."""
    logger.info("Generating Training Time plot...")
    plot_cfg = style_config.get('Plot', {})
    bar_cfg = style_config.get('BarChart', {})
    model_names = [m['name'] for m in all_models_data]
    colors = get_colors(model_names, style_config)

    time_data = {}
    for model_data in all_models_data:
        name = model_data['name']
        summary = model_data.get('summary', {})
        avg_time = summary.get('avg_epoch_time_s', None)
        if avg_time is not None:
            time_data[name] = avg_time
        else:
            # Fallback: Calculate from history if summary is missing
            history = model_data.get('history')
            if history is not None and not history.empty and 'epoch_time' in history.columns:
                 time_data[name] = history['epoch_time'].mean()
            else:
                 time_data[name] = np.nan

    # Prepare data for plotting
    plot_data = pd.DataFrame.from_dict(time_data, orient='index', columns=['avg_epoch_time_s'])
    plot_data = plot_data.reindex(model_names) # Keep original order
    plot_data['color'] = [colors.get(name, 'gray') for name in plot_data.index]

    fig, ax = plt.subplots(figsize=plot_cfg.get('figsize', [8, 5]))
    apply_plot_style(style_config)

    bars = ax.bar(plot_data.index, plot_data['avg_epoch_time_s'],
                  color=plot_data['color'], width=bar_cfg.get('width', 0.8))

    ax.set_xlabel('Model')
    ax.set_ylabel('Average Time per Epoch (seconds)')
    ax.set_title('Training Time Comparison')
    plt.xticks(rotation=45, ha='right')
    ax.grid(plot_cfg.get('grid', True), linestyle=plot_cfg.get('grid_style', '--'), alpha=plot_cfg.get('grid_alpha', 0.6), axis='y')

    # Add labels to bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if pd.notna(height):
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.1f}s', ha='center', va='bottom', fontsize=style_config.get('Font', {}).get('tick_size', 10))
        else:
             ax.text(bar.get_x() + bar.get_width() / 2., 0.1, 'N/A', ha='center', va='bottom', fontsize=style_config.get('Font', {}).get('tick_size', 10), color='red')

    if plot_data['avg_epoch_time_s'].notna().any():
         ax.set_ylim(bottom=0, top=plot_data['avg_epoch_time_s'].max() * 1.1)
    else:
         ax.set_ylim(bottom=0, top=1)

    plt.tight_layout()
    save_plot(fig, output_dir, 'comparison_training_time', style_config)
    plt.close(fig)


def generate_metrics_table(all_models_data, output_dir):
    """Generates a CSV and Markdown table summarizing key metrics."""
    logger.info("Generating final metrics summary table...")
    table_data = []

    for model_data in all_models_data:
        name = model_data['name']
        summary = model_data.get('summary', {})
        report_path = model_data['path'] / "results" / "best_model_classification_report.csv"
        report_df = None
        if report_path.exists():
             try:
                  report_df = pd.read_csv(report_path, index_col=0)
             except Exception as e:
                  logger.warning(f"Could not load classification report for {name}: {e}")

        row = {
            'Model': name,
            'Best Val Acc': summary.get('val_acc', summary.get('best_val_metric_value', np.nan)), # Prioritize final val_acc if available
            'Best Val AUC': summary.get('val_auc', np.nan),
            'Best Epoch': summary.get('best_epoch', np.nan),
            'Avg Epoch Time (s)': summary.get('avg_epoch_time_s', np.nan),
            'Trainable Params (M)': summary.get('trainable_params', np.nan) / 1_000_000 if pd.notna(summary.get('trainable_params')) else np.nan,
        }

        # Extract metrics from classification report if available
        if report_df is not None:
             row['Macro F1'] = report_df.loc['macro avg', 'f1-score'] if 'macro avg' in report_df.index else np.nan
             row['Macro Precision'] = report_df.loc['macro avg', 'precision'] if 'macro avg' in report_df.index else np.nan
             row['Macro Recall'] = report_df.loc['macro avg', 'recall'] if 'macro avg' in report_df.index else np.nan
             # Add weighted averages too
             row['Weighted F1'] = report_df.loc['weighted avg', 'f1-score'] if 'weighted avg' in report_df.index else np.nan
             row['Weighted Precision'] = report_df.loc['weighted avg', 'precision'] if 'weighted avg' in report_df.index else np.nan
             row['Weighted Recall'] = report_df.loc['weighted avg', 'recall'] if 'weighted avg' in report_df.index else np.nan
             # Add AUC if it was stored in the report
             if 'auc' in report_df.index:
                  row['Best Val AUC'] = report_df.loc['auc', 'precision'] # Stored in first column


        table_data.append(row)

    summary_df = pd.DataFrame(table_data)

    # Save as CSV
    csv_path = output_dir / "final_metrics_summary.csv"
    try:
        summary_df.to_csv(csv_path, index=False, float_format='%.4f')
        logger.info(f"Final metrics summary saved to {csv_path}")
    except Exception as e:
        logger.error(f"Error saving metrics summary CSV: {e}")

    # Save as Markdown
    md_path = output_dir / "final_metrics_summary.md"
    try:
        # Format floats for markdown
        md_df = summary_df.copy()
        float_cols = md_df.select_dtypes(include=['float']).columns
        for col in float_cols:
             md_df[col] = md_df[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
        # Format params column
        if 'Trainable Params (M)' in md_df.columns:
             # Check if the value is a number before formatting
             md_df['Trainable Params (M)'] = md_df['Trainable Params (M)'].map(
                 lambda x: f"{float(x):.1f}M" if pd.notna(x) and isinstance(x, (int, float, str)) and str(x) != "N/A" else "N/A"
             )


        md_string = md_df.to_markdown(index=False)
        with open(md_path, 'w') as f:
            f.write(md_string)
        logger.info(f"Final metrics summary saved to {md_path}")
    except Exception as e:
        logger.error(f"Error saving metrics summary Markdown: {e}")


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description='Generate visualizations for model training results.')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Path to the main output directory containing model subdirectories (e.g., ./output)')
    parser.add_argument('--config', type=str, default='style_config.yaml',
                        help='Path to the style configuration YAML file.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save generated plots (defaults to ./visualizations inside results_dir)')
    # parser.add_argument('--preset', type=str, default=None, help='Use a predefined style preset (e.g., nature, ieee)') # Future enhancement

    args = parser.parse_args()

    results_base_dir = Path(args.results_dir)
    vis_output_dir = Path(args.output_dir) if args.output_dir else results_base_dir / "visualizations"
    vis_output_dir.mkdir(parents=True, exist_ok=True)

    # Add file handler for logging in the visualization output directory
    log_file = vis_output_dir / 'visualization.log'
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    logger.info(f"Starting Visualization Generation")
    logger.info(f"Results Source Directory: {results_base_dir}")
    logger.info(f"Visualization Output Directory: {vis_output_dir}")
    logger.info(f"Style Config File: {args.config}")

    # Load style config
    style_config = load_style_config(args.config)

    # Find and load data for all models
    model_dirs = find_model_dirs(results_base_dir)
    if not model_dirs:
        sys.exit(1)

    all_models_data = [load_model_data(mdir) for mdir in model_dirs]
    # Filter out models where data loading failed completely (e.g., no history)
    all_models_data = [m for m in all_models_data if m.get('history') is not None or m.get('summary')]

    if not all_models_data:
         logger.error("No valid model data could be loaded. Exiting.")
         sys.exit(1)

    # --- Generate Plots ---
    plot_train_val_curves(all_models_data, style_config, vis_output_dir)
    plot_convergence_speed(all_models_data, style_config, vis_output_dir)
    plot_lr_schedule(all_models_data, style_config, vis_output_dir)
    plot_model_complexity(all_models_data, style_config, vis_output_dir)
    plot_training_time(all_models_data, style_config, vis_output_dir)

    # Individual plots
    for model_data in all_models_data:
        plot_confusion_matrix(model_data, style_config, vis_output_dir)

    # --- Generate Summary Table ---
    generate_metrics_table(all_models_data, vis_output_dir)

    logger.info(f"Visualization generation complete. Plots saved in: {vis_output_dir}")


if __name__ == "__main__":
    main()
