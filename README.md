# CharRNN Project

Генератор текста на символах (Character-Level RNN) на PyTorch.

## Установка
```bash
pip install -r requirements.txt
```

## Обучение
```bash
python train.py
```

## Генерация текста
```bash
python infer.py --prompt "Жил-был"
```

## Структура проекта
```
char_rnn_project/
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── trainer.py
│   ├── infer.py
│   └── utils.py
│
├── data/
├── checkpoints/
├── runs/
│
├── train.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Зависимости
- Python 3.10+
- PyTorch
- numpy
- tqdm
- tensorboard
