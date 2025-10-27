
import torch
import torch.nn.functional as F
import random

class TextGenerator:
    def __init__(self, model, stoi, itos, charset, device):
        self.model = model.to(device).eval()
        self.stoi = stoi
        self.itos = itos
        self.charset = charset
        self.device = device

    @torch.no_grad()
    def generate(self, max_len=300):
        context = torch.tensor([[self.stoi[random.choice(self.charset)]]], device=self.device)
        hidden = None
        generated = []
        for _ in range(max_len):
            logits, hidden = self.model(context, hidden)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_id = torch.multinomial(probs, 1)
            context = next_id
            generated.append(next_id.item())
        return ''.join([self.itos[i] for i in generated if i in self.itos])

