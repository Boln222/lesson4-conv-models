import argparse
import csv
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from models import FCNet, CNN, ResidualCNN
from utils import train_model, count_parameters, evaluate, save_curves, save_confusion_matrix


def loaders(dataset, batch_size, limit):
    root = Path('data')
    if dataset == 'mnist':
        data = datasets.MNIST
        transform = transforms.ToTensor()
    else:
        data = datasets.CIFAR10
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.5, .5, .5), (.5, .5, .5))])
    train, test = data(root, True, transform, download=True), data(root, False, transform, download=True)
    if limit:
        train, test = Subset(train, range(min(limit, len(train)))), Subset(test, range(min(limit // 5, len(test))))
    return DataLoader(train, batch_size, shuffle=True), DataLoader(test, batch_size), (1, 28) if dataset == 'mnist' else (3, 32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['mnist', 'cifar'], default='mnist')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    train, test, (channels, size) = loaders(args.dataset, args.batch_size, args.limit)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = {'FC': FCNet(channels, size), 'CNN': CNN(channels, depth=2), 'ResidualCNN': ResidualCNN(channels, blocks=2)}
    results, rows = {}, []
    out = Path('results') / f'{args.dataset}_comparison'
    out.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        history = train_model(model, train, test, device, args.epochs)
        accuracy, labels, predictions = evaluate(model, test, device)
        results[name] = history
        rows.append([name, round(accuracy, 4), count_parameters(model), round(history['seconds'], 2)])
        if name == 'ResidualCNN':
            save_confusion_matrix(labels, predictions, out / 'confusion_matrix.png')
    with open(out / 'metrics.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['model', 'test_accuracy', 'parameters', 'seconds'])
        writer.writerows(rows)
    save_curves(results, out / 'learning_curves.png')
    for row in rows:
        print(*row, sep=' | ')


if __name__ == '__main__':
    main()
