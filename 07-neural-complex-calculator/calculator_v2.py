# %% [markdown]
# # Neural-Network Complex-Number Calculator — v2 (improved)
# Changes over v1:
# 1. Larger network: 8 -> 256 -> 256 -> 128 -> 2 with BatchNorm (102,530 parameters)
# 2. More data: 500,000 samples (v1: 100k); |b| >= 0.5 for division
# 3. One-hot operation encoding (removes the false ordinal relation of v1's scalar code)
# 4. Standardised inputs AND outputs
# 5. Huber loss, cosine-annealed LR, gradient clipping, best-checkpoint restore; 150 epochs

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

OP_CODES = {"+": 0, "-": 1, "*": 2, "/": 3}
OP_SYMS = {"+": "+", "-": "−", "*": "×", "/": "÷"}


def exact_result(a_r, a_i, b_r, b_i, op):
    if op == "+":
        return a_r + b_r, a_i + b_i
    if op == "-":
        return a_r - b_r, a_i - b_i
    if op == "*":
        return a_r * b_r - a_i * b_i, a_r * b_i + a_i * b_r
    d = b_r ** 2 + b_i ** 2
    if d < 1e-12:
        raise ValueError("Division by zero")
    return (a_r * b_r + a_i * b_i) / d, (a_i * b_r - a_r * b_i) / d


def one_hot(op):
    oh = [0.0, 0.0, 0.0, 0.0]
    oh[OP_CODES[op]] = 1.0
    return oh


# %% Dataset: 500k samples, inputs and outputs standardised
def generate_dataset(n_samples=500_000, val_split=0.1):
    ops = list(OP_CODES.keys())
    X, Y = [], []
    for _ in range(n_samples):
        a_r, a_i = np.random.uniform(-10, 10, 2)
        b_r, b_i = np.random.uniform(-10, 10, 2)
        op = np.random.choice(ops)
        if op == "/":
            # keep |b| >= 0.5 so division is well-conditioned
            while np.hypot(b_r, b_i) < 0.5:
                b_r, b_i = np.random.uniform(-10, 10, 2)
        r_r, r_i = exact_result(a_r, a_i, b_r, b_i, op)
        X.append([a_r, a_i, b_r, b_i] + one_hot(op))   # 8-d input
        Y.append([r_r, r_i])

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    x_mean, x_std = X.mean(axis=0), X.std(axis=0) + 1e-8
    y_mean, y_std = Y.mean(axis=0), Y.std(axis=0) + 1e-8
    Xn, Yn = (X - x_mean) / x_std, (Y - y_mean) / y_std

    split = int(n_samples * (1 - val_split))
    return (torch.tensor(Xn[:split]), torch.tensor(Yn[:split]),
            torch.tensor(Xn[split:]), torch.tensor(Yn[split:]),
            x_mean, x_std, y_mean, y_std)


print("Generating 500,000 samples ...")
X_tr, Y_tr, X_val, Y_val, x_mean, x_std, y_mean, y_std = generate_dataset(500_000)
print(f"Train: {X_tr.shape[0]:,} | Val: {X_val.shape[0]:,} | in-dim {X_tr.shape[1]}")


# %% Model
class ComplexCalcNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(8, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.head = nn.Linear(128, 2)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(self.trunk(x))


model = ComplexCalcNet().to(device)
print(model)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")


# %% Training: Adam + Huber + cosine annealing + grad clipping, 150 epochs
def train(model, X_tr, Y_tr, X_val, Y_val, epochs=150, batch_size=1024, lr=3e-3):
    train_loader = DataLoader(TensorDataset(X_tr.to(device), Y_tr.to(device)),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val.to(device), Y_val.to(device)),
                            batch_size=2048)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.HuberLoss(delta=1.0)   # less sensitive to large division errors than MSE

    train_losses, val_losses = [], []
    best_val, best_state = float("inf"), None
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running += loss.item() * xb.size(0)
        t_loss = running / len(train_loader.dataset)

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                v_loss += criterion(model(xb), yb).item() * xb.size(0)
        v_loss /= len(val_loader.dataset)

        scheduler.step()
        train_losses.append(t_loss)
        val_losses.append(v_loss)
        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 15 == 0 or epoch == 1:
            print(f"Epoch {epoch:>3}/{epochs} | train {t_loss:.6f} | val {v_loss:.6f} | "
                  f"lr {optimizer.param_groups[0]['lr']:.2e}")

    model.load_state_dict(best_state)
    print(f"Restored best model (val loss = {best_val:.6f})")
    return train_losses, val_losses


train_losses, val_losses = train(model, X_tr, Y_tr, X_val, Y_val, epochs=150)

# %% Loss curve
plt.figure(figsize=(9, 4))
plt.plot(train_losses, label="Train (Huber)", linewidth=1.8)
plt.plot(val_losses, label="Val (Huber)", linewidth=1.8, linestyle="--")
plt.yscale("log"); plt.xlabel("Epoch"); plt.ylabel("Huber loss (log)")
plt.title("Training & Validation Loss — v2"); plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig("loss_curve_v2.png", dpi=150); plt.show()


# %% Inference (de-normalise outputs)
def nn_predict(a_r, a_i, b_r, b_i, op):
    raw = np.array([[a_r, a_i, b_r, b_i] + one_hot(op)], dtype=np.float32)
    norm = (raw - x_mean) / x_std
    model.eval()
    with torch.no_grad():
        pred_norm = model(torch.tensor(norm).to(device)).cpu().numpy()[0]
    pred = pred_norm * y_std + y_mean
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


# %% Same 10 examples as v1 for a direct comparison
EXAMPLES = [
    (3, 2, 1, -4, "+"), (5, -3, 2, 7, "+"),
    (4, 1, 4, -1, "-"), (0, 5, 0, -5, "-"),
    (2, 3, 1, 4, "*"), (3, -2, 3, 2, "*"),
    (1, 1, 1, -1, "/"), (10, 5, 2, 1, "/"),
    (0, 1, 0, 1, "*"), (7, -7, 3, 4, "/"),
]

print("=" * 65 + "\n 10 EXAMPLE PREDICTIONS (v2)\n" + "=" * 65)
rows = []
for idx, (ar, ai, br, bi, op) in enumerate(EXAMPLES, 1):
    print(f"── Example {idx} " + "─" * 40)
    rows.append((idx, ar, ai, br, bi, op, *evaluate(ar, ai, br, bi, op)))
