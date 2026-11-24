import argparse
import csv
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from models import CNN, ResidualCNN, kernel_model
from utils import train_model, count_parameters, save_curves


def get_loaders(limit, batch_size):
    transform = transforms.ToTensor()
    train = datasets.MNIST('data', True, transform, download=True)
    test = datasets.MNIST('data', False, transform, download=True)
    if limit:
        train, test = Subset(train, range(min(limit, len(train)))), Subset(test, range(min(limit // 5, len(test))))
    return DataLoader(train, batch_size, shuffle=True), DataLoader(test, batch_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    train, test = get_loaders(args.limit, args.batch_size)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = {'kernel_3': kernel_model('3'), 'kernel_5': kernel_model('5'), 'kernel_7': kernel_model('7'), 'kernel_mixed': kernel_model('mixed'), 'depth_2': CNN(depth=2), 'depth_4': CNN(depth=4), 'depth_6': CNN(depth=6), 'residual': ResidualCNN(blocks=3)}
    results, rows = {}, []
    for name, model in models.items():
        result = train_model(model, train, test, device, args.epochs)
        results[name] = result
        rows.append([name, round(result['test_acc'][-1], 4), count_parameters(model), round(result['seconds'], 2), round(sum(result['gradients'].values()) / max(1, len(result['gradients'])), 6)])
    out = Path('results/architecture_analysis')
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'metrics.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['model', 'test_accuracy', 'parameters', 'seconds', 'mean_gradient_norm'])
        writer.writerows(rows)
    save_curves(results, out / 'learning_curves.png')
    for row in rows:
        print(*row, sep=' | ')


if __name__ == '__main__':
    main()
