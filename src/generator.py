import torch
import torch.nn.functional as F
import random
import numpy as np

class TextGenerator:
    def __init__(self, model, stoi, itos, charset, device):
        self.model = model.to(device).eval()
        self.stoi = stoi
        self.itos = itos
        self.charset = charset
        self.device = device

    def _apply_temperature(self, logits, temperature=1.0):
        """Применяет temperature scaling к логитам"""
        if temperature == 0:
            return logits
        return logits / temperature

    def _top_k_filtering(self, logits, top_k=0):
        """Фильтрует логиты, оставляя только top_k токенов"""
        if top_k > 0:
            values, _ = torch.topk(logits, top_k)
            min_value = values[:, -1].unsqueeze(1)
            logits = torch.where(logits < min_value, 
                               torch.ones_like(logits) * -float('inf'), 
                               logits)
        return logits

    def _top_p_filtering(self, logits, top_p=0.0):
        """Nucleus sampling - фильтрует логиты по накопленной вероятности"""
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Убираем токены с накопленной вероятностью выше top_p
            sorted_indices_to_remove = cumulative_probs > top_p
            # Оставляем первый токен (даже если он превышает top_p)
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = -float('inf')
        return logits

    @torch.no_grad()
    def generate(self, prompt=None, max_len=300, temperature=1.0, top_k=0, top_p=0.0):
        """
        Генерирует текст с различными стратегиями семплирования
        
        Args:
            prompt: начальный текст (если None, используется случайный символ)
            max_len: максимальная длина генерируемого текста
            temperature: температура для контроля случайности (0.1-2.0)
            top_k: количество топ токенов для фильтрации (0 = отключено)
            top_p: nucleus sampling threshold (0.0-1.0, 0 = отключено)
        """
        # Инициализация контекста
        if prompt is not None and prompt.strip():
            # Используем промпт, фильтруем неизвестные символы
            filtered_prompt = ''.join(ch for ch in prompt if ch in self.stoi)
            if filtered_prompt:
                context_ids = [self.stoi[ch] for ch in filtered_prompt]
                context = torch.tensor([context_ids], device=self.device)
            else:
                # Если в промпте нет известных символов, используем случайный
                context = torch.tensor([[self.stoi[random.choice(self.charset)]]], device=self.device)
        else:
            # Случайный начальный символ
            context = torch.tensor([[self.stoi[random.choice(self.charset)]]], device=self.device)

        hidden = None
        generated = []
        
        # Генерируем по одному токену
        for _ in range(max_len):
            logits, hidden = self.model(context, hidden)
            logits = logits[:, -1, :]  # Берем только последний токен
            
            # Применяем различные стратегии семплирования
            logits = self._apply_temperature(logits, temperature)
            logits = self._top_k_filtering(logits, top_k)
            logits = self._top_p_filtering(logits, top_p)
            
            # Семплируем следующий токен
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            
            # Обновляем контекст для следующей итерации
            context = next_id
            generated.append(next_id.item())

        # Декодируем результат
        result = ''.join([self.itos[i] for i in generated if i in self.itos])
        
        # Добавляем промпт в начало, если он был
        if prompt is not None and prompt.strip():
            filtered_prompt = ''.join(ch for ch in prompt if ch in self.stoi)
            if filtered_prompt:
                result = filtered_prompt + result
                
        return result

    @torch.no_grad()
    def generate_samples(self, num_samples=1, **kwargs):
        """Генерирует несколько образцов текста"""
        samples = []
        for i in range(num_samples):
            sample = self.generate(**kwargs)
            samples.append(f"=== Образец {i+1} ===\n{sample}\n")
        return "\n".join(samples)

