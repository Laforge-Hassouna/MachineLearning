import cv2
import sys
import os
import time
import glob
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

def process_video_with_model(video_path, model_path):
    """Process a video with a YOLO model and return inference times."""
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    # Read video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output video writer
    output_filename = f"results/{os.path.basename(video_path)}_{os.path.basename(os.path.dirname(model_path))}_results.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    inference_times = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Start timer
        start_time = time.time()

        # Run inference
        results = model.predict(frame)

        # End timer and store duration
        end_time = time.time()
        inference_times.append(end_time - start_time)

        # Plot and save frame
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

    # Release resources
    cap.release()
    out.release()

    return inference_times

def calculate_stats(times):
    """Calculate statistics from inference times."""
    if not times:
        return None
    stats = {
        "mean": sum(times) / len(times),
        "worst": max(times),
        "best": min(times),
        "total": sum(times),
        "count": len(times),
    }
    return stats

def plot_stats(model_names, stats, stat_name, filename):
    """Plot a bar chart for a specific statistic and save it."""
    values = [stats[name][stat_name] for name in model_names]
    colors = ["red" if name in ["yolo11n_img1024_ep20", "fs_yolo11m_img384_ep20"] else "skyblue" for name in model_names]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, values, color=colors)

    # Highlight with star markers and set label
    for i, name in enumerate(model_names):
        if name in ["yolo11n_img1024_ep20", "fs_yolo11m_img384_ep20"]:
            plt.scatter(i, values[i], color="gold", marker="*", s=200, zorder=3)

    plt.xlabel("Modèles")
    plt.ylabel(f"{stat_name} (secondes)")
    plt.title(f"Comparaison des {stat_name} par modèle")
    plt.xticks(rotation=45)

    # Add legend only if there are highlighted models
    if any(name in ["yolo11n_img1024_ep20", "fs_yolo11m_img384_ep20"] for name in model_names):
        plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py <video_path> <models_directory>")
        sys.exit(1)

    video_path = sys.argv[1]
    models_dir = sys.argv[2]

    # Create results directory
    os.makedirs("results", exist_ok=True)

    # Find all best.pt files in subdirectories
    model_paths = glob.glob(os.path.join(models_dir, "*", "weights", "best.pt"))

    if not model_paths:
        print("Aucun fichier best.pt trouvé dans les sous-dossiers.")
        sys.exit(1)

    all_stats = {}

    for model_path in model_paths:
        model_name = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
        inference_times = process_video_with_model(video_path, model_path)
        stats = calculate_stats(inference_times)
        all_stats[model_name] = stats

    # Print all statistics at the end
    print("\n--- Statistiques globales ---")
    for model_name, stats in all_stats.items():
        print(f"\nModèle: {model_name}")
        print(f"  Moyenne: {stats['mean']:.4f} secondes")
        print(f"  Pire: {stats['worst']:.4f} secondes")
        print(f"  Meilleur: {stats['best']:.4f} secondes")
        print(f"  Total: {stats['total']:.4f} secondes")
        print(f"  Nombre d'inférences: {stats['count']}")

    # Plot all statistics
    model_names = list(all_stats.keys())
    plot_stats(model_names, all_stats, "mean", "results/comparaison_moyenne.png")
    plot_stats(model_names, all_stats, "worst", "results/comparaison_pire.png")
    plot_stats(model_names, all_stats, "best", "results/comparaison_meilleur.png")
    plot_stats(model_names, all_stats, "total", "results/comparaison_total.png")

if __name__ == "__main__":
    main()
