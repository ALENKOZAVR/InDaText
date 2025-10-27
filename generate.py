#!/usr/bin/env python3
"""
Скрипт для генерации текста с помощью обученной модели CharRNN
"""

import torch
import argparse
import os
import sys

# Добавляем src в путь для импортов
sys.path.append('src')

from src.model import CharRNN
from src.infer import TextGenerator
from src.utils import load_text_and_charset
from src.config import load_config


def load_model_and_generator(config_path="config.json"):
    """Загружает модель и создает генератор текста"""
    
    # Загружаем конфигурацию
    config = load_config(config_path)
    
    # Определяем устройство
    if config.system['device'] == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.system['device']
    
    print(f"Используется устройство: {device}")
    
    # Загружаем данные для получения словарей
    _, _, charset, stoi, itos, vocab_size, _ = load_text_and_charset(
        text_path=config.data['text_path'],
        charset_path=config.data['charset_path']
    )
    
    # Создаем модель
    model = CharRNN(
        vocab_size=vocab_size,
        hidden_size=config.model['hidden_size'],
        num_layers=config.model['num_layers'],
        dropout=config.model['dropout']
    ).to(device)
    
    # Загружаем чекпоинт
    checkpoint_path = config.paths['checkpoint_path']
    if not os.path.exists(checkpoint_path):
        print(f"❌ Чекпоинт не найден: {checkpoint_path}")
        print("Сначала запустите обучение: python train.py")
        sys.exit(1)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint.get("epoch", "неизвестно")
        print(f"✅ Загружена модель с эпохи {epoch}")
    except Exception as e:
        print(f"❌ Ошибка загрузки чекпоинта: {e}")
        sys.exit(1)
    
    # Создаем генератор
    generator = TextGenerator(model, stoi, itos, charset, device)
    
    return generator, config


def main():
    parser = argparse.ArgumentParser(description="Генерация текста с CharRNN")
    
    parser.add_argument("--prompt", type=str, default="", 
                       help="Начальный текст для генерации")
    parser.add_argument("--length", type=int, default=300,
                       help="Длина генерируемого текста (по умолчанию: 300)")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Температура для контроля случайности (0.1-2.0, по умолчанию: 1.0)")
    parser.add_argument("--top_k", type=int, default=0,
                       help="Top-k фильтрация (0 = отключено, по умолчанию: 0)")
    parser.add_argument("--top_p", type=float, default=0.0,
                       help="Top-p nucleus sampling (0.0-1.0, 0 = отключено)")
    parser.add_argument("--samples", type=int, default=1,
                       help="Количество образцов для генерации (по умолчанию: 1)")
    parser.add_argument("--config", type=str, default="config.json",
                       help="Путь к файлу конфигурации")
    
    # Предустановленные режимы
    parser.add_argument("--creative", action="store_true",
                       help="Креативный режим (temperature=1.3, top_p=0.9)")
    parser.add_argument("--conservative", action="store_true", 
                       help="Консервативный режим (temperature=0.7, top_k=50)")
    parser.add_argument("--balanced", action="store_true",
                       help="Сбалансированный режим (temperature=0.9, top_k=100)")
    
    args = parser.parse_args()
    
    # Применяем предустановленные режимы
    if args.creative:
        args.temperature = 1.3
        args.top_p = 0.9
        args.top_k = 0
        print("🎨 Креативный режим активирован")
    elif args.conservative:
        args.temperature = 0.7
        args.top_k = 50
        args.top_p = 0.0
        print("🎯 Консервативный режим активирован")
    elif args.balanced:
        args.temperature = 0.9
        args.top_k = 100
        args.top_p = 0.0
        print("⚖️ Сбалансированный режим активирован")
    
    # Валидация параметров
    if args.temperature <= 0:
        print("❌ Температура должна быть больше 0")
        sys.exit(1)
    
    if args.top_p < 0 or args.top_p > 1:
        print("❌ top_p должно быть между 0.0 и 1.0")
        sys.exit(1)
    
    # Загружаем модель и генератор
    print("⏳ Загружаем модель...")
    generator, config = load_model_and_generator(args.config)
    
    # Выводим параметры генерации
    print("\n" + "="*60)
    print("ПАРАМЕТРЫ ГЕНЕРАЦИИ:")
    print(f"📝 Промпт: '{args.prompt}' (пустой = случайный старт)")
    print(f"📏 Длина: {args.length}")
    print(f"🌡️ Температура: {args.temperature}")
    if args.top_k > 0:
        print(f"🔝 Top-k: {args.top_k}")
    if args.top_p > 0:
        print(f"🎯 Top-p: {args.top_p}")
    print(f"🎲 Количество образцов: {args.samples}")
    print("="*60)
    
    # Генерируем текст
    if args.samples == 1:
        print("\n🚀 Генерация текста...")
        result = generator.generate(
            prompt=args.prompt,
            max_len=args.length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
        print("\n📖 РЕЗУЛЬТАТ:")
        print("-" * 40)
        print(result)
        print("-" * 40)
    else:
        print(f"\n🚀 Генерация {args.samples} образцов...")
        for i in range(args.samples):
            result = generator.generate(
                prompt=args.prompt,
                max_len=args.length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p
            )
            print(f"\n📖 ОБРАЗЕЦ {i+1}:")
            print("-" * 40)
            print(result)
            print("-" * 40)
    
    print("\n✅ Генерация завершена!")


if __name__ == "__main__":
    main()