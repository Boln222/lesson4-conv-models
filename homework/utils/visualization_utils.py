from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def save_curves(results, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, item in results.items():
        epochs = range(1, len(item['loss']) + 1)
        axes[0].plot(epochs, item['loss'], label=name)
        axes[1].plot(epochs, item['test_acc'], label=name)
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
    if Path(path).stem.startswith('confusion_') and Path(path).stem != 'confusion_matrix':
        names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        plt.xticks(range(10), names, rotation=45, ha='right')
        plt.yticks(range(10), names)
        plt.title(Path(path).stem.removeprefix('confusion_'))
        np.savetxt(Path(path).with_suffix('.csv'), matrix, fmt='%d', delimiter=',')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_cifar_diagnostics(results, out):
    fig, axes = plt.subplots(2, len(results), figsize=(16, 7), layout='constrained')
    for col, (name, h) in enumerate(results.items()):
        top, bottom = axes[:, col]
        epochs = range(1, len(h['loss']) + 1)
        top.plot(epochs, h['train_acc'], label='train')
        top.plot(epochs, h['test_acc'], label='test')
        top.set(title=name, xlabel='Epoch', ylabel='Accuracy', ylim=(0, 1))
        top.legend()
        layers = list(h['gradient_epochs'][0])
        for i, norms in enumerate(h['gradient_epochs'], 1):
            bottom.plot(range(1, len(layers) + 1), list(norms.values()), marker='.', label=str(i))
        bottom.set(xlabel='Weight layer (input to output)', ylabel='Mean gradient norm', yscale='log')
        bottom.legend(title='Epoch', fontsize=7, ncol=2)
    fig.savefig(out / 'accuracy_and_gradients.png')
    plt.close(fig)
