import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def load_results(filepath):
    """Load results.csv into a DataFrame."""
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def plot_metrics(df1, df2, model1_name, model2_name, output_dir="plots"):
    """Plot comparison of metrics for two models."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    metrics = [
        'metrics/precision(B)',
        'metrics/recall(B)',
        'metrics/mAP50(B)',
        'metrics/mAP50-95(B)'
    ]

    plt.figure(figsize=(12, 8))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
        if metric in df1.columns and metric in df2.columns:
            plt.plot(df1['epoch'], df1[metric], label=f'{model1_name}', color='blue')
            plt.plot(df2['epoch'], df2[metric], label=f'{model2_name}', color='red')
        else:
            print(f"Metric {metric} not found in one or both DataFrames.")
            continue
        plt.title(metric)
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/metrics_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()

def plot_losses(df1, df2, model1_name, model2_name, output_dir="plots"):
    """Plot comparison of training losses for two models."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    losses = [
        'train/box_loss',
        'train/cls_loss',
        'train/dfl_loss'
    ]

    plt.figure(figsize=(12, 8))
    for i, loss in enumerate(losses, 1):
        plt.subplot(2, 2, i)
        if loss in df1.columns and loss in df2.columns:
            plt.plot(df1['epoch'], df1[loss], label=f'{model1_name}', color='blue')
            plt.plot(df2['epoch'], df2[loss], label=f'{model2_name}', color='red')
        else:
            print(f"Loss {loss} not found in one or both DataFrames.")
            continue
        plt.title(loss)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/losses_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(model1_cm, model2_cm, model1_name, model2_name, output_dir="plots"):
    """Plot confusion matrices for two models."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.heatmap(model1_cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model1_name}')

    plt.subplot(1, 2, 2)
    sns.heatmap(model2_cm, annot=True, fmt='d', cmap='Reds')
    plt.title(f'Confusion Matrix - {model2_name}')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrices.png', dpi=200, bbox_inches='tight')
    plt.close()

def main():
    # Load results for both models
    model1_path = './general_models/fs_yolo11m_img384_ep20/results.csv'
    model2_path = './general_models/yolo11n_img1024_ep20/results.csv'
    model1_name = 'Model fs_yolo11 backbone m 384p'
    model2_name = 'Model yolo11 backbone n 1024p'

    df1 = load_results(model1_path)
    df2 = load_results(model2_path)

    if df1 is None or df2 is None:
        print("Failed to load one or both results files.")
        return

    # Plot metrics and losses
    plot_metrics(df1, df2, model1_name, model2_name)
    plot_losses(df1, df2, model1_name, model2_name)

    # Example confusion matrices (replace with actual data)
    model1_cm = np.array([[50, 10], [5, 35]])
    model2_cm = np.array([[45, 15], [10, 30]])
    plot_confusion_matrix(model1_cm, model2_cm, model1_name, model2_name)

if __name__ == '__main__':
    main()