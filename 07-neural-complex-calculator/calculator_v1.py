# %% [markdown]
# # Neural-Network Complex-Number Calculator — v1 (baseline)
# A small MLP learns (a, b, op) -> a op b for complex a, b
# and op in {+, -, x, /}. Input: [a_r, a_i, b_r, b_i, op_code] (5-d); output: [r_r, r_i].

# %%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Scalar operation code (v1). v2 replaces this with a one-hot vector.
OP_CODES = {"+": 0.25, "-": 0.50, "*": 0.75, "/": 1.00}
OP_SYMS = {"+": "+", "-": "−", "*": "×", "/": "÷"}


# %% Exact arithmetic (ground truth)
def exact_result(a_r, a_i, b_r, b_i, op):
    if op == "+":
        return a_r + b_r, a_i + b_i
    if op == "-":
        return a_r - b_r, a_i - b_i
    if op == "*":
        return a_r * b_r - a_i * b_i, a_r * b_i + a_i * b_r
    d = b_r ** 2 + b_i ** 2
    if d < 1e-12:
        raise ValueError("Division by zero (b ≈ 0)")
    return (a_r * b_r + a_i * b_i) / d, (a_i * b_r - a_r * b_i) / d


# %% Dataset generation
def generate_dataset(n_samples: int = 100_000, val_split: float = 0.1):
    ops = list(OP_CODES.keys())
    X, Y = [], []
    for _ in range(n_samples):
        a_r, a_i = np.random.uniform(-10, 10, 2)
        b_r, b_i = np.random.uniform(-10, 10, 2)
        op = np.random.choice(ops)
        # avoid a near-zero denominator for division
        if op == "/":
            while abs(b_r) < 0.1 and abs(b_i) < 0.1:
                b_r, b_i = np.random.uniform(-10, 10, 2)
        r_r, r_i = exact_result(a_r, a_i, b_r, b_i, op)
        X.append([a_r, a_i, b_r, b_i, OP_CODES[op]])
        Y.append([r_r, r_i])

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)

    # min-max normalise inputs to [-1, 1]
    x_min, x_max = X.min(axis=0), X.max(axis=0)
    X_norm = 2 * (X - x_min) / (x_max - x_min + 1e-8) - 1

    split = int(n_samples * (1 - val_split))
    return (torch.tensor(X_norm[:split]), torch.tensor(Y[:split]),
            torch.tensor(X_norm[split:]), torch.tensor(Y[split:]),
            x_min, x_max)


X_tr, Y_tr, X_val, Y_val, x_min, x_max = generate_dataset(100_000)
print(f"Train: {X_tr.shape[0]:,} | Val: {X_val.shape[0]:,} | "
      f"in {tuple(X_tr.shape)} -> out {tuple(Y_tr.shape)}")


# %% Model: 5 -> 64 -> 64 -> 2 MLP (4,674 parameters)
class ComplexCalcNet(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


model = ComplexCalcNet(hidden=64).to(device)
print(model)
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters()):,}")


# %% Training: Adam + MSE + ReduceLROnPlateau, 60 epochs
def train(model, X_tr, Y_tr, X_val, Y_val, epochs=60, batch_size=512, lr=1e-3):
    train_loader = DataLoader(TensorDataset(X_tr.to(device), Y_tr.to(device)),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val.to(device), Y_val.to(device)),
                            batch_size=1024)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
        train_loss = running / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                val_loss += criterion(model(xb), yb).item() * xb.size(0)
        val_loss /= len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:>3}/{epochs} | train MSE {train_loss:.6f} | val MSE {val_loss:.6f}")
    return train_losses, val_losses


train_losses, val_losses = train(model, X_tr, Y_tr, X_val, Y_val, epochs=60)

# %% Loss curve
plt.figure(figsize=(9, 4))
plt.plot(train_losses, label="Train MSE", linewidth=1.8)
plt.plot(val_losses, label="Val MSE", linewidth=1.8, linestyle="--")
plt.yscale("log"); plt.xlabel("Epoch"); plt.ylabel("MSE loss (log scale)")
plt.title("Training & Validation Loss"); plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig("loss_curve_v1.png", dpi=150); plt.show()


# %% Inference helpers
def nn_predict(a_r, a_i, b_r, b_i, op):
    raw = np.array([[a_r, a_i, b_r, b_i, OP_CODES[op]]], dtype=np.float32)
    norm = 2 * (raw - x_min) / (x_max - x_min + 1e-8) - 1
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(norm).to(device)).cpu().numpy()[0]
    return float(pred[0]), float(pred[1])


def fmt_complex(r, i, prec=4):
    sign = "+" if i >= 0 else "−"
    return f"{r:+.{prec}f} {sign} {abs(i):.{prec}f}i"


def evaluate(a_r, a_i, b_r, b_i, op, verbose=True):
    nn_r, nn_i = nn_predict(a_r, a_i, b_r, b_i, op)
    ex_r, ex_i = exact_result(a_r, a_i, b_r, b_i, op)
    abs_err = np.hypot(nn_r - ex_r, nn_i - ex_i)
    magnitude = np.hypot(ex_r, ex_i) or 1.0
    rel_pct = abs_err / magnitude * 100
    if verbose:
        print(f" ({fmt_complex(a_r, a_i)}) {OP_SYMS[op]} ({fmt_complex(b_r, b_i)})")
        print(f"  NN prediction : {fmt_complex(nn_r, nn_i)}")
        print(f"  Exact result  : {fmt_complex(ex_r, ex_i)}")
        print(f"  |error|       : {abs_err:.6f} ({rel_pct:.4f}% relative)\n")
    return nn_r, nn_i, ex_r, ex_i, abs_err, rel_pct


# %% Ten test examples (same set is used for v2)
EXAMPLES = [
    (3, 2, 1, -4, "+"),   # 1  addition
    (5, -3, 2, 7, "+"),   # 2  addition
    (4, 1, 4, -1, "-"),   # 3  subtraction
    (0, 5, 0, -5, "-"),   # 4  subtraction (pure imaginary)
    (2, 3, 1, 4, "*"),    # 5  multiplication
    (3, -2, 3, 2, "*"),   # 6  multiplication (conjugate pair)
    (1, 1, 1, -1, "/"),   # 7  division
    (10, 5, 2, 1, "/"),   # 8  division
    (0, 1, 0, 1, "*"),    # 9  i × i = −1
    (7, -7, 3, 4, "/"),   # 10 division (mixed signs)
]

print("=" * 65 + "\n 10 EXAMPLE PREDICTIONS (v1)\n" + "=" * 65)
rows = []
for idx, (ar, ai, br, bi, op) in enumerate(EXAMPLES, 1):
    print(f"── Example {idx} " + "─" * 40)
    rows.append((idx, ar, ai, br, bi, op, *evaluate(ar, ai, br, bi, op)))
