from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def save_curves(results, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, item in results.items():
        axes[0].plot(item['loss'], label=name)
        axes[1].plot(item['test_acc'], label=name)
    axes[0].set_title('Loss')
    axes[1].set_title('Test accuracy')
    for axis in axes:
        axis.set_xlabel('Epoch')
        axis.legend()
        axis.grid()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_confusion_matrix(labels, predictions, path, classes=10):
    matrix = np.zeros((classes, classes), dtype=int)
    for y, p in zip(labels, predictions):
        matrix[y, p] += 1
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap='Blues')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
