# Лабораторная работа: CNN и кастомные слои

## Запуск

Из папки `homework`:

```powershell
pip install torch torchvision matplotlib
python homework_cnn_vs_fc_comparison.py --dataset mnist --epochs 3
python homework_cnn_vs_fc_comparison.py --dataset cifar --limit 5000 --batch-size 64 --epochs 6
python homework_cnn_architecture_analysis.py --epochs 3
python homework_custom_layers_experiments.py
```

Для быстрой проверки без долгого обучения добавь `--limit 1000`.

Результаты сохраняются в `results/`: таблицы CSV, графики обучения и confusion matrix.

Для задания 1.2 эта команда берёт первые 5 000 обучающих и 1 000 тестовых изображений CIFAR-10. Без `--limit` используются все 50 000 и 10 000. Сравниваются глубокая FCNet и две одинаковые ResidualCNN, у второй включён Dropout2d(0.2). Итоги и графики — в [отчёте по CIFAR-10](results/cifar_comparison/REPORT.md).

Условия всех трёх заданий — в [HOMEWORK.md](../HOMEWORK.md).
