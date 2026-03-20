import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.nn import functional as F
from model.gpt import GPTLanguageModel, block_size, device

# Training parameters
batch_size = 32
learning_rate = 1e-3
max_iters = 2000
eval_interval = 200

# Load Math dataset
with open("data/math/train.txt", "r") as f:
    train_text = f.read()

with open("data/math/test.txt", "r") as f:
    test_text = f.read()

chars = sorted(list(set(train_text + test_text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return "".join([itos[i] for i in l])

train_data = torch.tensor(encode(train_text), dtype=torch.long)
test_data = torch.tensor(encode(test_text), dtype=torch.long)

# Batch loader
def get_batch(data):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# Model
model = GPTLanguageModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for step in range(max_iters):
    xb, yb = get_batch(train_data)
    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % eval_interval == 0:
        print(f"step {step}, loss {loss.item():.4f}")

def test(expr):
    model.eval()
    idx = torch.tensor([encode(expr)], dtype=torch.long).to(device)
    out = model.generate(idx, max_new_tokens=6)
    generated = decode(out[0].tolist()[len(expr):]).split("\n")[0]
    print(expr + generated)

torch.save(model.state_dict(), "model/boolean_gpt.pt")

print("\nSanity checks:")

test("47+38=")   
test("82-32=")   
test("70+35=")   
test("28-11=")   
test("58+99=")   

torch.save(model.state_dict(), "model_weights_part1.pth")
