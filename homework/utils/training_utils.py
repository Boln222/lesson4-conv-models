import time
import logging
import torch
import torch.nn as nn


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    labels, predictions = [], []
    with torch.no_grad():
        for x, y in loader:
            out = model(x.to(device))
            pred = out.argmax(1).cpu()
            correct += (pred == y).sum().item()
            total += len(y)
            labels.extend(y.tolist())
            predictions.extend(pred.tolist())
    return correct / total, labels, predictions


def gradient_norms(model):
    return {name: p.grad.norm().item() for name, p in model.named_parameters() if p.grad is not None}


def train_model(model, train_loader, test_loader, device, epochs=3, lr=1e-3, train_eval=None):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    history = {'loss': [], 'train_acc': [], 'test_acc': [], 'seconds': 0, 'gradients': {}}
    started = time.perf_counter()
    history['train_seconds'] = 0
    history['gradient_epochs'] = []
    for _ in range(epochs):
        epoch_start = time.perf_counter()
        norms = {}
        model.train()
        correct = total = loss_sum = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            history['gradients'] = gradient_norms(model)
            if train_eval is not None:
                for name, value in history['gradients'].items():
                    if name.endswith('weight') and model.get_parameter(name).ndim > 1:
                        norms[name] = norms.get(name, 0) + value
            optimizer.step()
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
        history['train_seconds'] += time.perf_counter() - epoch_start
        history['gradient_epochs'].append({k: v / len(train_loader) for k, v in norms.items()})
        test_acc, _, _ = evaluate(model, test_loader, device)
        history['loss'].append(loss_sum / total)
        train_acc = correct / total if train_eval is None else evaluate(model, train_eval, device)[0]
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        logging.info('Epoch %d/%d | train %.1f%% | test %.1f%%', len(history['loss']), epochs, 100 * train_acc, 100 * test_acc)
    history['seconds'] = time.perf_counter() - started
    return history
