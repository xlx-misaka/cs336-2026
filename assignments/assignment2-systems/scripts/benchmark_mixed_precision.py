from torch import nn
import torch.nn.functional as F
import torch


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        h1 = self.fc1(x)
        h1a = self.relu(h1)
        h2 = self.ln(h1a)
        logits = self.fc2(h2)
        return h1, h1a, h2, logits


device = "cuda"
model = ToyModel(32, 5).to(device)

x = torch.randn(8, 32, device=device)
y = torch.randint(0, 5, (8,), device=device)

model.zero_grad(set_to_none=True)

with torch.autocast(device_type=device, dtype=torch.bfloat16):
    h1, h1a, h2, logits = model(x)
    loss = F.cross_entropy(logits, y)

loss.backward()

print("param fc1.weight :", model.fc1.weight.dtype)
print("param ln.weight  :", model.ln.weight.dtype)
print("param ln.bias    :", model.ln.bias.dtype)
print("param fc2.weight :", model.fc2.weight.dtype)

print("fc1 output       :", h1.dtype)
print("relu output      :", h1a.dtype)
print("layernorm output :", h2.dtype)
print("logits           :", logits.dtype)
print("loss             :", loss.dtype)

assert model.fc1.weight.grad is not None
print("grad fc1.weight  :", model.fc1.weight.grad.dtype)
assert model.ln.weight.grad is not None
print("grad ln.weight   :", model.ln.weight.grad.dtype)
assert model.ln.bias.grad is not None
print("grad ln.bias     :", model.ln.bias.grad.dtype)
assert model.fc2.weight.grad is not None
print("grad fc2.weight  :", model.fc2.weight.grad.dtype)
