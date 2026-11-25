import argparse
import csv
import json
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from models import FCNet, CNN, ResidualCNN
from utils import train_model, count_parameters, evaluate, save_curves, save_confusion_matrix
from utils.visualization_utils import save_cifar_diagnostics


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
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    train, test, (channels, size) = loaders(args.dataset, args.batch_size, args.limit)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = {'FC': FCNet(channels, size), 'CNN': CNN(channels, depth=2), 'ResidualCNN': ResidualCNN(channels, blocks=2)}
    cifar = args.dataset == 'cifar'
    if cifar:
        torch.manual_seed(42)
        base = ResidualCNN(3)
        regularized = ResidualCNN(3, regularized=True)
        regularized.load_state_dict(base.state_dict())
        models = {'DeepFC': FCNet(3, 32, hidden=(512, 256, 128, 64)),
                  'ResidualCNN': base, 'ResidualDropout': regularized}
    train_eval = DataLoader(train.dataset, args.batch_size) if cifar else None
    results, rows = {}, []
    out = Path('results') / f'{args.dataset}_comparison'
    out.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        logging.info('%s', name)
        if cifar:
            torch.manual_seed(42)
            train.sampler.generator = torch.Generator().manual_seed(42)
        history = train_model(model, train, test, device, args.epochs, train_eval=train_eval)
        accuracy, labels, predictions = evaluate(model, test, device)
        results[name] = history
        rows.append([name, round(accuracy, 4), count_parameters(model), round(history['seconds'], 2)])
        if cifar:
            rows[-1] += [history['train_acc'][-1], round(history['train_seconds'], 2)]
            save_confusion_matrix(labels, predictions, out / f'confusion_{name}.png')
        elif name == 'ResidualCNN':
            save_confusion_matrix(labels, predictions, out / 'confusion_matrix.png')
    with open(out / 'metrics.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        header = ['model', 'test_accuracy', 'parameters', 'seconds']
        writer.writerow(header + (['train_accuracy', 'training_seconds'] if cifar else []))
        writer.writerows(rows)
    save_curves(results, out / 'learning_curves.png')
    if cifar:
        save_cifar_diagnostics(results, out)
        (out / 'history.json').write_text(json.dumps(results, indent=2))
    for row in rows:
        print(*row, sep=' | ')


if __name__ == '__main__':
    main()
