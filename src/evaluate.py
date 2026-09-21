import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from src.dataset import CLASS_NAMES, load_and_preprocess_data

def evaluate_model(model_path="model/fashion_cnn_model.keras",
                   history_path="model/model_history.json",
                   output_metrics_path="model/test_metrics.json",
                   plots_dir="plots"):
    """
    Evaluates trained model on the 10,000 Fashion-MNIST test samples,
    computes detailed metrics, and produces diagnostic plots.
    """
    os.makedirs("model", exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print(f"Loading trained model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    print("Loading test dataset...")
    _, _, (x_test, y_test) = load_and_preprocess_data()

    print("Evaluating model performance on test set...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, batch_size=64, verbose=1)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    print("Generating predictions...")
    y_pred_probs = model.predict(x_test, batch_size=64, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Classification report
    report_dict = classification_report(
        y_test, y_pred, target_names=CLASS_NAMES, output_dict=True
    )
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Save metrics JSON
    metrics_summary = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "macro_avg_f1": float(report_dict["macro avg"]["f1-score"]),
        "weighted_avg_f1": float(report_dict["weighted avg"]["f1-score"]),
        "per_class": {
            cls: {
                "precision": float(report_dict[cls]["precision"]),
                "recall": float(report_dict[cls]["recall"]),
                "f1-score": float(report_dict[cls]["f1-score"]),
                "support": int(report_dict[cls]["support"])
            }
            for cls in CLASS_NAMES
        },
        "confusion_matrix": cm.tolist()
    }

    with open(output_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"Saved evaluation metrics to {output_metrics_path}")

    # Plot 1: Confusion Matrix Heatmap
    plt.figure(figsize=(10, 8), dpi=150)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar=True
    )
    plt.title(f"Fashion-MNIST Confusion Matrix (Test Accuracy: {test_accuracy * 100:.2f}%)", fontsize=14, pad=15)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_plot_path = os.path.join(plots_dir, "confusion_matrix.png")
    plt.savefig(cm_plot_path)
    plt.close()
    print(f"Saved confusion matrix plot to {cm_plot_path}")

    # Plot 2: Per-Class Accuracy / F1 Bar Chart
    plt.figure(figsize=(10, 5), dpi=150)
    f1_scores = [report_dict[cls]["f1-score"] * 100 for cls in CLASS_NAMES]
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(CLASS_NAMES)))
    bars = plt.bar(CLASS_NAMES, f1_scores, color=colors, edgecolor='black', linewidth=0.8)
    plt.axhline(test_accuracy * 100, color='red', linestyle='--', linewidth=1.5,
                label=f'Overall Accuracy ({test_accuracy * 100:.1f}%)')
    plt.title("Per-Class F1-Score (%) on Test Set", fontsize=13, pad=12)
    plt.xlabel("Fashion Item Class", fontsize=11)
    plt.ylabel("F1-Score (%)", fontsize=11)
    plt.ylim(70, 102)
    plt.xticks(rotation=40, ha="right")
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    plt.legend(loc='lower right')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height + 0.8,
                 f"{height:.1f}%", ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    class_dist_path = os.path.join(plots_dir, "class_distribution.png")
    plt.savefig(class_dist_path)
    plt.close()
    print(f"Saved class accuracy plot to {class_dist_path}")

    # Plot 3: Training and Validation Curves
    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)

        epochs = range(1, len(history.get("accuracy", [])) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

        # Accuracy
        axes[0].plot(epochs, [a * 100 for a in history["accuracy"]], 'o-', label='Train Accuracy', color='#2b5c8f')
        axes[0].plot(epochs, [a * 100 for a in history["val_accuracy"]], 's--', label='Val Accuracy', color='#d95f02')
        axes[0].set_title("Training vs Validation Accuracy", fontsize=13)
        axes[0].set_xlabel("Epoch", fontsize=11)
        axes[0].set_ylabel("Accuracy (%)", fontsize=11)
        axes[0].grid(True, linestyle=':', alpha=0.6)
        axes[0].legend(loc="lower right")

        # Loss
        axes[1].plot(epochs, history["loss"], 'o-', label='Train Loss', color='#2b5c8f')
        axes[1].plot(epochs, history["val_loss"], 's--', label='Val Loss', color='#e7298a')
        axes[1].set_title("Training vs Validation Loss", fontsize=13)
        axes[1].set_xlabel("Epoch", fontsize=11)
        axes[1].set_ylabel("Loss", fontsize=11)
        axes[1].grid(True, linestyle=':', alpha=0.6)
        axes[1].legend(loc="upper right")

        plt.suptitle("Fashion-MNIST CNN Convergence with Regularization", fontsize=15, y=1.02)
        plt.tight_layout()
        curves_path = os.path.join(plots_dir, "training_curves.png")
        plt.savefig(curves_path)
        plt.close()
        print(f"Saved training curves plot to {curves_path}")

    return metrics_summary

if __name__ == '__main__':
    evaluate_model()
