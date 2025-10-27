import torch

def load_text_and_charset(text_path="Дореформенная_одной_строкой.txt", charset_path="charset.txt"):
    # === ПОДГОТОВКА ТЕКСТА ===
    text = open(text_path, encoding="utf-8").read()

    # === ЗАГРУЗКА ЧАРСЕТА ===
    with open(charset_path, encoding="utf-8") as f:
        charset = [line.strip("\n") for line in f if line.strip("\n") != ""]

    # === СЛОВАРИ stoi / itos ПО ЧАРСЕТУ ===
    stoi = {ch: i for i, ch in enumerate(charset)}
    itos = {i: ch for ch, i in stoi.items()}

    # === ОЧИСТКА ТЕКСТА: оставляем только символы, есть в чарсете ===
    text = ''.join(ch for ch in text if ch in charset)

    # === ФУНКЦИИ КОДИРОВКИ / ДЕКОДИРОВКИ ===
    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda l: ''.join([itos[i] for i in l if i in itos])

    data = torch.tensor(encode(text), dtype=torch.long)
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]
    vocab_size = len(charset)

    return train_data, val_data, charset, stoi, itos, vocab_size, decode
