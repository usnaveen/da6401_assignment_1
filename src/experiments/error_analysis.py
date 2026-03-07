"""
Section 2.8 — Error Analysis (5 marks)
Confusion matrix + creative visualization for best model.

Run from project root:
    cd src && python -m experiments.error_analysis
"""
import numpy as np
import wandb
import json
from argparse import Namespace
from ann.neural_network import NeuralNetwork
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_loader import get_artifact_path, load_dataset


def load_best_model():
    """Load the best model and its config."""
    with open(get_artifact_path("best_config.json"), "r") as f:
        config = json.load(f)
    args = Namespace(**config)

    model = NeuralNetwork(args)
    weights = np.load(get_artifact_path("best_model.npy"), allow_pickle=True).item()
    model.set_weights(weights)
    return model, args


def run_error_analysis():
    model, args = load_best_model()

    # Load test data
    if args.dataset == "mnist":
        _, (X_test_flat, y_test) = load_dataset("mnist")
        class_names = [str(i) for i in range(10)]
    else:
        _, (X_test_flat, y_test) = load_dataset("fashion_mnist")
        class_names = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        ]

    X_test = X_test_flat.reshape(-1, 28, 28)

    wandb.init(
        project="da6401-assignment-1",
        name="error_analysis",
        tags=["error_analysis"],
        config=vars(args),
    )

    # Get predictions
    logits = model.forward(X_test_flat)
    y_pred = np.argmax(logits, axis=1)

    # === 1. Confusion Matrix ===
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix — Best Model')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    wandb.log({"confusion_matrix": wandb.Image("confusion_matrix.png")})
    print("Confusion matrix logged.")

    # Also log as W&B confusion matrix
    wandb.log({
        "confusion_matrix_interactive": wandb.plot.confusion_matrix(
            probs=None, y_true=y_test.tolist(), preds=y_pred.tolist(),
            class_names=class_names
        )
    })

    # === 2. Creative Visualization: Most Confused Pairs ===
    # Find top misclassified pairs
    misclassified = np.where(y_pred != y_test)[0]
    print(f"Total misclassified: {len(misclassified)} / {len(y_test)}")

    # Find the most common confusion pairs
    from collections import Counter
    confusion_pairs = Counter()
    for idx in misclassified:
        confusion_pairs[(y_test[idx], y_pred[idx])] += 1

    top_pairs = confusion_pairs.most_common(5)
    print("Top 5 confused pairs:")
    for (true_cls, pred_cls), count in top_pairs:
        print(f"  True: {class_names[true_cls]} -> Predicted: {class_names[pred_cls]} ({count} times)")

    # Visualize examples of the top confused pair
    if top_pairs:
        true_cls, pred_cls = top_pairs[0][0]
        confused_indices = [
            idx for idx in misclassified
            if y_test[idx] == true_cls and y_pred[idx] == pred_cls
        ][:10]

        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        fig.suptitle(
            f'Most Confused: True={class_names[true_cls]} → Predicted={class_names[pred_cls]}',
            fontsize=14
        )
        for i, ax in enumerate(axes.flat):
            if i < len(confused_indices):
                idx = confused_indices[i]
                ax.imshow(X_test[idx], cmap='gray')
                conf = float(np.max(np.exp(logits[idx]) / np.sum(np.exp(logits[idx]))))
                ax.set_title(f'conf={conf:.2f}', fontsize=9)
            ax.axis('off')
        plt.tight_layout()
        plt.savefig("most_confused_examples.png", dpi=150)
        wandb.log({"most_confused_examples": wandb.Image("most_confused_examples.png")})
        print("Most confused examples logged.")

    # === 3. Per-class accuracy bar chart ===
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#e74c3c' if acc < 0.9 else '#2ecc71' for acc in per_class_acc]
    bars = ax.bar(class_names, per_class_acc, color=colors)
    ax.set_ylabel('Accuracy')
    ax.set_title('Per-Class Accuracy')
    ax.set_ylim(0, 1.05)
    for bar, acc in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.2f}', ha='center', va='bottom', fontsize=9)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("per_class_accuracy.png", dpi=150)
    wandb.log({"per_class_accuracy": wandb.Image("per_class_accuracy.png")})
    print("Per-class accuracy logged.")

    wandb.finish()
    print("\nError analysis complete! Check your W&B project.")


if __name__ == "__main__":
    run_error_analysis()
