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
python infer.py --help

Примеры:
python infer.py --prompt "Жил-был" --length 200 --temperature 0.8
python infer.py --prompt "Добрый день" --creative --length 150
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
