
import torch
import torch.nn.functional as F
import math, time, os, random

class Trainer:
    def __init__(self, model, optimizer, train_data, val_data, vocab_size, writer, device, checkpoint_path, charset, stoi, itos, decode, config=None):
        self.model = model
        self.optimizer = optimizer
        self.train_data = train_data
        self.val_data = val_data
        self.vocab_size = vocab_size
        self.writer = writer
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.charset = charset
        self.stoi = stoi
        self.itos = itos
        self.decode = decode
        self.config = config

    def topk_accuracy(self, logits, targets, k=1):
        topk = logits.topk(k, dim=-1).indices
        correct = topk.eq(targets.unsqueeze(-1)).any(dim=-1)
        return correct.float().mean().item()

    def get_batch(self, split="train", block_size=50, batch_size=64):
        data = self.train_data if split == "train" else self.val_data
        
        # Проверяем, что данных достаточно для создания батча
        if len(data) <= block_size:
            raise ValueError(f"Недостаточно данных: len(data)={len(data)}, block_size={block_size}")
        
        # Генерируем случайные индексы так, чтобы y не выходил за границы
        max_start_idx = len(data) - block_size - 1
        ix = torch.randint(0, max_start_idx, (batch_size,))
        
        x = torch.stack([data[i:i + block_size] for i in ix])
        y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
        return x.to(self.device), y.to(self.device)

    def save_checkpoint(self, epoch):
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        tmp_path = self.checkpoint_path + ".tmp"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, tmp_path)
        os.replace(tmp_path, self.checkpoint_path)
        print(f"\U0001F4BE Чекпоинт сохранён (эпоха {epoch})")

    @torch.no_grad()
    def sample_and_log(self, epoch):
        # Идентичная логика генерации как в karp.py
        context = torch.tensor([[self.stoi[random.choice(self.charset)]]], device=self.device)
        hidden = None
        generated = []
        for _ in range(300):
            logits, hidden = self.model(context, hidden)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_id = torch.multinomial(probs, 1)
            context = next_id
            generated.append(next_id.item())
        sample_text = self.decode(generated)
        self.writer.add_text("Samples/generated", f"**Epoch {epoch}**\n\n{sample_text}", epoch)
        print("Пример генерации:\n", sample_text[:400])
        print("=" * 80)

    def train(self, num_epochs=35000, block_size=50, batch_size=64, log_interval=100):
        start_epoch = 0
        if os.path.exists(self.checkpoint_path):
            try:
                checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                start_epoch = checkpoint["epoch"] + 1
                print(f"✅ Загружено сохранение с эпохи {start_epoch}")
            except Exception as e:
                print(f"⚠️ Не удалось загрузить чекпоинт ({e}). Начинаем с нуля.")
        else:
            print("🆕 Начинаем обучение с нуля")

        for epoch in range(start_epoch, num_epochs):
            start_time = time.time()

            # TRAIN
            self.model.train()
            x, y = self.get_batch("train", block_size, batch_size)
            logits, _ = self.model(x)
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), y.view(-1))
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping для стабильности обучения RNN
            if self.config and self.config.training.get('gradient_clip_norm'):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.training['gradient_clip_norm'])
            
            self.optimizer.step()

            self.writer.add_scalar("Loss/train", loss.item(), epoch)
            self.writer.add_scalar("LR", self.optimizer.param_groups[0]["lr"], epoch)

            # VALIDATION
            self.model.eval()
            with torch.no_grad():
                val_x, val_y = self.get_batch("val", block_size, batch_size)
                val_logits, _ = self.model(val_x)
                val_loss = F.cross_entropy(val_logits.view(-1, self.vocab_size), val_y.view(-1))
                top1 = self.topk_accuracy(val_logits, val_y, k=1)
                top5 = self.topk_accuracy(val_logits, val_y, k=5)
                val_perplexity = math.exp(val_loss.item())

                self.writer.add_scalar("Loss/val", val_loss.item(), epoch)
                self.writer.add_scalar("Perplexity/val", val_perplexity, epoch)
                self.writer.add_scalar("Top1/val", top1, epoch)
                self.writer.add_scalar("Top5/val", top5, epoch)

            if epoch % log_interval == 0:
                print(
                    f"Epoch {epoch:04d} | Train={loss.item():.4f} | Val={val_loss.item():.4f} "
                    f"| PPL={val_perplexity:.2f} | Top1={top1:.3f} | Top5={top5:.3f} "
                    f"| time={time.time() - start_time:.2f}s"
                )
                self.sample_and_log(epoch)
                self.save_checkpoint(epoch)

        self.writer.flush()
        print("✅ Обучение завершено.")
