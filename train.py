import torch
from torch.utils.tensorboard import SummaryWriter

from src.model import CharRNN
from src.training import Trainer
from src.utils import load_text_and_charset
from src.config import load_config

if __name__ == "__main__":
    # Загружаем конфигурацию
    config = load_config("config.json")
    
    # Устанавливаем seed для воспроизводимости
    if config.system['seed'] is not None:
        torch.manual_seed(config.system['seed'])
    
    # Определяем устройство
    if config.system['device'] == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.system['device']

    # Загружаем данные используя конфигурацию
    train_data, val_data, charset, stoi, itos, vocab_size, decode = load_text_and_charset(
        text_path=config.data['text_path'],
        charset_path=config.data['charset_path']
    )

    # Создаем модель с параметрами из конфигурации
    model = CharRNN(
        vocab_size=vocab_size,
        hidden_size=config.model['hidden_size'],
        num_layers=config.model['num_layers'],
        dropout=config.model['dropout']
    ).to(device)
    
    # Создаем оптимизатор с параметрами из конфигурации
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.training['lr'],
        weight_decay=config.training['weight_decay']
    )

    # Создаем директории из конфигурации
    writer = SummaryWriter(config.paths['tensorboard_dir'])
    checkpoint_path = config.paths['checkpoint_path']

    print(f"Обучение запущено на {device}. Параметров: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Конфигурация загружена из config.json")

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_data=train_data,
        val_data=val_data,
        vocab_size=vocab_size,
        writer=writer,
        device=device,
        checkpoint_path=checkpoint_path,
        charset=charset,
        stoi=stoi,
        itos=itos,
        decode=decode,
        config=config  # Передаем конфигурацию в тренер
    )

    # Используем параметры из конфигурации
    trainer.train(
        num_epochs=config.training['num_epochs'],
        block_size=config.training['block_size'], 
        batch_size=config.training['batch_size'],
        log_interval=config.training['log_interval']
    )

