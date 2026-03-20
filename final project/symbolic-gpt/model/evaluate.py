import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.gpt import GPTLanguageModel, block_size, device


# Load Math dataset
with open("data/math/train.txt", "r") as f:
    train_text = f.read()

with open("data/math/test.txt", "r") as f:
    test_lines = f.readlines()

chars = sorted(list(set(train_text + "".join(test_lines))))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return "".join([itos[i] for i in l])

# Load Math GPT model
vocab_size = len(chars)
model = GPTLanguageModel(vocab_size).to(device)

model.load_state_dict(torch.load("model_weights_part1.pth", map_location=device))
model.eval()

# Greedy generation
@torch.no_grad()
def generate_greedy(model, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

# Evaluation
correct = 0
total = 0
errors = []

for line in test_lines:
    expr = line.strip()
    if "=" not in expr:
        continue

    prompt, target = expr.split("=")
    prompt = prompt + "="

    idx = torch.tensor([encode(prompt)], dtype=torch.long).to(device)

    out = generate_greedy(model, idx, max_new_tokens=6)

    prediction = decode(out[0].tolist()[len(prompt):]).split("\n")[0]

    if prediction == target:
        correct += 1
    else:
        errors.append((prompt, prediction, target))

    total += 1

accuracy = correct / total

print(f"Exact-match accuracy: {accuracy:.3f} ({correct}/{total})")

print("\nSample errors:")
for e in errors[:5]:
    print(f"{e[0]} → predicted {e[1]}, expected {e[2]}")
