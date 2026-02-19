"""
Group Assignment 1: Incremental Learning on Large Data

Feed-forward neural network with 3 sigmoid hidden layers,
incremental mini-batch training from CSV chunks,
learning curve, variable importance, and partial dependence plots.
"""

import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_SEED = int(os.environ.get("ILNN_SEED", "42"))
np.random.seed(BASE_SEED)

FEATURE_COLS = ["sku", "price", "order", "duration", "category"]
TARGET_COL = "quantity"
FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_COLS)}

# Training hyperparameters (pre-tuned)
BATCH_SIZE = 64
EPOCHS = 12
CHUNK_ROWS = 60000
HIDDEN_SIZES = (160, 80, 40)
LEARNING_RATE = 7.5e-4
X_CLIP = 7.0
WEIGHT_DECAY = 1e-6
LR_DECAY = 0.996
MIN_LR = 1e-5
SKU_HASH_BUCKETS = 32
CATEGORY_HASH_BUCKETS = 16
SKU_SCALE = 0.0
CATEGORY_SCALE = 0.55
SHUFFLE_BUFFER_CHUNKS = 8


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def get_memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def moving_average(values, window=200):
    if len(values) < window:
        return np.array(values, dtype=np.float64)
    arr = np.asarray(values, dtype=np.float64)
    return np.convolve(arr, np.ones(window) / window, mode="valid")


def compute_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 0.0 if ss_tot <= 1e-12 else 1.0 - ss_res / ss_tot


def infer_header(path):
    with open(path, "r", encoding="utf-8") as f:
        tokens = [t.strip().lower() for t in f.readline().strip().split(",")]
    expected = ["sku", "price", "quantity", "order", "duration", "category"]
    return tokens == expected


def resolve_paths():
    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
    def find(name):
        for d in [script_dir, os.getcwd(), os.path.join(os.getcwd(), "Deep Learning")]:
            p = os.path.join(d, name)
            if os.path.exists(p):
                return p
        return os.path.join(script_dir, name)
    return find("pricing.csv"), find("pricing_test.csv"), script_dir


def count_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        n = sum(1 for _ in f)
    return max(0, n - (1 if infer_header(path) else 0))


# ---------------------------------------------------------------------------
# Feature Engineering (log_shift_diff mode only)
# ---------------------------------------------------------------------------
def transform_features(x_raw):
    """Log-shifted differences + hashed one-hot for sku/category + interactions."""
    x = np.asarray(x_raw, dtype=np.float64)
    n = x.shape[0]

    sku = np.mod(x[:, FEATURE_INDEX["sku"]].astype(np.int64), SKU_HASH_BUCKETS)
    cat = np.mod(x[:, FEATURE_INDEX["category"]].astype(np.int64), CATEGORY_HASH_BUCKETS)

    price = np.log1p(np.maximum(x[:, FEATURE_INDEX["price"]], 0.0))
    order = np.log1p(np.maximum(x[:, FEATURE_INDEX["order"]], 0.0))
    duration = np.log1p(np.maximum(x[:, FEATURE_INDEX["duration"]], 0.0))

    eps = 1e-3
    numeric = np.column_stack([
        price, order, duration,
        order - duration,               # diff_od
        price - order,                   # diff_po
        price - duration,                # diff_pd
        order / (duration + eps),        # ratio_od
        price * duration,                # interactions
        order * duration,
        price * order,
    ])

    cat_oh = np.zeros((n, CATEGORY_HASH_BUCKETS), dtype=np.float64)
    cat_oh[np.arange(n), cat] = 1.0
    sku_oh = np.zeros((n, SKU_HASH_BUCKETS), dtype=np.float64)
    sku_oh[np.arange(n), sku] = 1.0

    return np.concatenate([numeric, cat_oh, sku_oh], axis=1)


NUM_NUMERIC = 10  # 7 base + 3 interactions


# ---------------------------------------------------------------------------
# Standardization
# ---------------------------------------------------------------------------
def compute_stats(train_file, chunksize=50000):
    """Streaming mean/std for features and target."""
    n, sx, sx2, sy, sy2 = 0, None, None, 0.0, 0.0
    for chunk in pd.read_csv(train_file, chunksize=chunksize):
        xf = transform_features(chunk[FEATURE_COLS].values)
        y = chunk[TARGET_COL].values.astype(np.float64)
        if sx is None:
            sx = np.zeros(xf.shape[1], dtype=np.float64)
            sx2 = np.zeros(xf.shape[1], dtype=np.float64)
        n += len(chunk)
        sx += xf.sum(axis=0)
        sx2 += (xf * xf).sum(axis=0)
        sy += y.sum()
        sy2 += (y * y).sum()
    if n == 0:
        raise ValueError("Empty training file.")
    x_mean = sx / n
    x_std = np.sqrt(np.maximum(sx2 / n - x_mean ** 2, 1e-12))
    y_mean = sy / n
    y_std = max(np.sqrt(sy2 / n - y_mean ** 2), 1e-12)
    return x_mean.astype(np.float32), x_std.astype(np.float32), np.float32(y_mean), np.float32(y_std)


def preprocess_x(x_raw, x_mean, x_std):
    """Transform, standardize, clip, and apply category/sku scaling."""
    xf = transform_features(x_raw)
    z = np.clip((xf - x_mean) / x_std, -X_CLIP, X_CLIP)
    cat_start = NUM_NUMERIC
    cat_end = cat_start + CATEGORY_HASH_BUCKETS
    z[:, cat_start:cat_end] *= CATEGORY_SCALE
    z[:, cat_end:] *= SKU_SCALE
    return z


def std_y(y, y_mean, y_std):
    return (y.astype(np.float64) - y_mean) / y_std


def unstd_y(y_std_val, y_mean, y_std):
    return y_std_val * y_std + y_mean


# ---------------------------------------------------------------------------
# Neural Network
# ---------------------------------------------------------------------------
class IncrementalNeuralNetwork:
    def __init__(self, input_size, hidden_sizes=HIDDEN_SIZES, lr=LEARNING_RATE):
        self.lr = lr
        self.grad_clip = 5.0
        self.weight_decay = WEIGHT_DECAY

        # Xavier init
        sizes = [input_size] + list(hidden_sizes) + [1]
        self.layers = []
        for i in range(len(sizes) - 1):
            lim = np.sqrt(6.0 / (sizes[i] + sizes[i + 1]))
            W = np.random.uniform(-lim, lim, (sizes[i], sizes[i + 1])).astype(np.float32)
            b = np.zeros((1, sizes[i + 1]), dtype=np.float32)
            self.layers.append({"W": W, "b": b})

        # Adam state
        self.mw = [np.zeros_like(l["W"]) for l in self.layers]
        self.vw = [np.zeros_like(l["W"]) for l in self.layers]
        self.mb = [np.zeros_like(l["b"]) for l in self.layers]
        self.vb = [np.zeros_like(l["b"]) for l in self.layers]
        self.t = 0

        self.mse_history = []
        self.instances_history = []
        self.instances_trained = 0

    @staticmethod
    def sigmoid(x):
        x = np.clip(x, -40, 40)
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, x):
        acts = [x]
        cur = x
        for i, layer in enumerate(self.layers):
            z = cur @ layer["W"] + layer["b"]
            cur = z if i == len(self.layers) - 1 else self.sigmoid(z)
            acts.append(cur)
        return cur, acts

    def train_batch(self, x, y):
        out, acts = self.forward(x)
        err = out - y.reshape(-1, 1)
        mse = float(np.mean(err ** 2))
        delta = err
        self.t += 1

        for i in range(len(self.layers) - 1, -1, -1):
            W_old = self.layers[i]["W"]
            dW = (acts[i].T @ delta) / x.shape[0] + self.weight_decay * W_old
            db = np.mean(delta, axis=0, keepdims=True)
            dW = np.clip(dW, -self.grad_clip, self.grad_clip)
            db = np.clip(db, -self.grad_clip, self.grad_clip)

            # Adam update
            self.mw[i] = 0.9 * self.mw[i] + 0.1 * dW
            self.vw[i] = 0.999 * self.vw[i] + 0.001 * dW * dW
            self.mb[i] = 0.9 * self.mb[i] + 0.1 * db
            self.vb[i] = 0.999 * self.vb[i] + 0.001 * db * db

            mw_h = self.mw[i] / (1 - 0.9 ** self.t)
            vw_h = self.vw[i] / (1 - 0.999 ** self.t)
            mb_h = self.mb[i] / (1 - 0.9 ** self.t)
            vb_h = self.vb[i] / (1 - 0.999 ** self.t)

            self.layers[i]["W"] = W_old - self.lr * mw_h / (np.sqrt(vw_h) + 1e-8)
            self.layers[i]["b"] -= self.lr * mb_h / (np.sqrt(vb_h) + 1e-8)

            if i > 0:
                delta = (delta @ W_old.T) * (acts[i] * (1.0 - acts[i]))

        self.instances_trained += x.shape[0]
        self.instances_history.append(self.instances_trained)
        self.mse_history.append(mse)
        return mse

    def predict(self, x):
        cur = x
        for i, layer in enumerate(self.layers):
            z = cur @ layer["W"] + layer["b"]
            cur = z if i == len(self.layers) - 1 else self.sigmoid(z)
        return cur.flatten()


# ---------------------------------------------------------------------------
# Training (incremental, chunked)
# ---------------------------------------------------------------------------
def train_incremental(model, train_file, x_mean, x_std, y_mean, y_std):
    print(f"\n{'=' * 50}\nTRAINING — {EPOCHS} epochs, batch={BATCH_SIZE}\n{'=' * 50}")
    start = time.time()
    mem = [get_memory_mb()]
    initial_lr = model.lr

    for epoch in range(1, EPOCHS + 1):
        model.lr = max(initial_lr * (LR_DECAY ** (epoch - 1)), MIN_LR)
        rng = np.random.RandomState(BASE_SEED + epoch)
        buffer = []
        batch_count = 0

        def process_chunk(chunk):
            nonlocal batch_count
            xp = preprocess_x(chunk[FEATURE_COLS].values.astype(np.float32), x_mean, x_std)
            yp = std_y(chunk[TARGET_COL].values, y_mean, y_std)
            for j in range(0, len(xp), BATCH_SIZE):
                model.train_batch(xp[j:j + BATCH_SIZE], yp[j:j + BATCH_SIZE])
                batch_count += 1

        for chunk in pd.read_csv(train_file, chunksize=CHUNK_ROWS):
            chunk = chunk.sample(frac=1.0, random_state=rng.randint(0, 2**31 - 1))
            buffer.append(chunk)
            if len(buffer) >= SHUFFLE_BUFFER_CHUNKS:
                process_chunk(buffer.pop(rng.randint(0, len(buffer))))
        while buffer:
            process_chunk(buffer.pop(rng.randint(0, len(buffer))))

        mem.append(get_memory_mb())
        print(f"  Epoch {epoch}/{EPOCHS}  lr={model.lr:.6f}  "
              f"instances={model.instances_trained}  mem={mem[-1]:.0f}MB")

    elapsed = time.time() - start
    print(f"Training time: {elapsed:.1f}s | Max memory: {max(mem):.0f}MB")
    return {"training_time": elapsed, "max_memory": max(mem), "avg_memory": np.mean(mem)}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def load_test_data(test_file):
    if infer_header(test_file):
        df = pd.read_csv(test_file)
    else:
        df = pd.read_csv(test_file, header=None,
                         names=["sku", "price", "quantity", "order", "duration", "category"])
    return df[FEATURE_COLS].values.astype(np.float32), df[TARGET_COL].values.astype(np.float32)


def align_test_scales(x_test, train_file):
    """Rescale test features if median ratio vs train is wildly off."""
    train_med = pd.read_csv(train_file, usecols=FEATURE_COLS).median().values
    x = x_test.copy()
    for i, name in enumerate(FEATURE_COLS):
        tr = abs(float(train_med[i]))
        if tr < 1e-8:
            continue
        te = abs(float(np.median(x[:, i]))) + 1e-12
        ratio = te / (tr + 1e-12)
        if ratio > 50 or ratio < 0.02:
            x[:, i] /= ratio
            print(f"  Rescaled test feature '{name}' by {ratio:.4g}")
    return x


def evaluate(model, test_file, train_file, x_mean, x_std, y_mean, y_std):
    print(f"\n{'=' * 50}\nEVALUATION\n{'=' * 50}")
    x_test, y_test = load_test_data(test_file)
    x_test = align_test_scales(x_test, train_file)
    y_pred = unstd_y(model.predict(preprocess_x(x_test, x_mean, x_std)), y_mean, y_std)

    mse = float(np.mean((y_test - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    r2 = float(compute_r2(y_test, y_pred))
    print(f"  MSE:  {mse:.4f}\n  RMSE: {rmse:.4f}\n  R²:   {r2:.4f}")
    return {"mse": mse, "rmse": rmse, "r2": r2, "y_true": y_test, "y_pred": y_pred}


# ---------------------------------------------------------------------------
# Interpretability
# ---------------------------------------------------------------------------
def sample_training_rows(train_file, n=5000):
    parts, kept = [], 0
    for chunk in pd.read_csv(train_file, chunksize=20000):
        if kept >= n:
            break
        take = min(n - kept, len(chunk))
        parts.append(chunk.sample(n=take, random_state=BASE_SEED))
        kept += take
    df = pd.concat(parts, ignore_index=True)
    return df[FEATURE_COLS].values.astype(np.float32), df[TARGET_COL].values.astype(np.float32)


def permutation_importance(model, x_raw, y_std_vals, x_mean, x_std, repeats=5):
    x_pp = preprocess_x(x_raw, x_mean, x_std)
    base_mse = float(np.mean((model.predict(x_pp) - y_std_vals) ** 2))
    result = {}
    for i, name in enumerate(FEATURE_COLS):
        deltas = []
        for _ in range(repeats):
            x_perm = x_raw.copy()
            x_perm[:, i] = x_perm[np.random.permutation(len(x_perm)), i]
            pred = model.predict(preprocess_x(x_perm, x_mean, x_std))
            deltas.append(float(np.mean((pred - y_std_vals) ** 2)) - base_mse)
        result[name] = float(np.mean(deltas))
    return result


def partial_dependence(model, x_raw, x_mean, x_std, y_mean, y_std, grid_points=40):
    curves = {}
    for i, name in enumerate(FEATURE_COLS):
        grid = np.linspace(x_raw[:, i].min(), x_raw[:, i].max(), grid_points)
        means = []
        for v in grid:
            x_tmp = x_raw.copy()
            x_tmp[:, i] = v
            pred = unstd_y(model.predict(preprocess_x(x_tmp, x_mean, x_std)), y_mean, y_std)
            means.append(float(pred.mean()))
        curves[name] = (grid, np.array(means))
    return curves


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_learning_curve(model, output_path, window=200):
    ma = moving_average(model.mse_history, window)
    x = model.instances_history[window - 1:] if len(model.mse_history) >= window else model.instances_history
    plt.figure(figsize=(12, 6))
    plt.plot(x, ma, linewidth=1.0)
    plt.xlabel("Instances Learned")
    plt.ylabel(f"Moving Avg MSE (window={window})")
    plt.title("Learning Curve")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_importance(importances, output_path):
    pairs = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
    plt.figure(figsize=(9, 5))
    plt.barh(range(len(pairs)), [v for _, v in pairs])
    plt.yticks(range(len(pairs)), [k for k, _ in pairs])
    plt.gca().invert_yaxis()
    plt.xlabel("Importance (ΔMSE)")
    plt.title("Permutation Variable Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_partial_dependence(curves, output_path):
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for i, name in enumerate(FEATURE_COLS):
        ax = axes.flatten()[i]
        grid, yvals = curves[name]
        ax.plot(grid, yvals, linewidth=1.8)
        ax.set_title(name)
        ax.set_xlabel(name)
        ax.set_ylabel("Predicted quantity")
        ax.grid(alpha=0.3)
    axes.flatten()[-1].set_visible(False)  # hide 6th subplot
    plt.suptitle("Partial Dependence Plots", y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 50)
    print("Incremental Learning on Large Data")
    print("=" * 50)

    train_file, test_file, output_dir = resolve_paths()
    for f in [train_file, test_file]:
        if not os.path.exists(f):
            raise FileNotFoundError(f)

    # Compute streaming stats
    print("Computing feature/target statistics...")
    x_mean, x_std, y_mean, y_std = compute_stats(train_file)
    input_size = len(x_mean)
    print(f"Input dimension: {input_size}")
    print(f"Target mean/std: {y_mean:.4f} / {y_std:.4f}")

    # Build and train model
    print(f"\nArchitecture: {input_size} -> {HIDDEN_SIZES} -> 1 (sigmoid hidden, linear output)")
    model = IncrementalNeuralNetwork(input_size)
    train_stats = train_incremental(model, train_file, x_mean, x_std, y_mean, y_std)

    # Evaluate
    eval_results = evaluate(model, test_file, train_file, x_mean, x_std, y_mean, y_std)

    # Interpretability plots
    print(f"\n{'=' * 50}\nGENERATING PLOTS\n{'=' * 50}")
    x_sample, y_sample = sample_training_rows(train_file)
    y_std_sample = std_y(y_sample, y_mean, y_std)

    plot_learning_curve(model, os.path.join(output_dir, "learning_curve.png"))

    print("Computing variable importance...")
    imp = permutation_importance(model, x_sample, y_std_sample, x_mean, x_std)
    plot_importance(imp, os.path.join(output_dir, "variable_importance.png"))

    print("Computing partial dependence...")
    curves = partial_dependence(model, x_sample, x_mean, x_std, y_mean, y_std)
    plot_partial_dependence(curves, os.path.join(output_dir, "partial_dependence.png"))

    # Summary
    print(f"\n{'=' * 50}\nSUMMARY\n{'=' * 50}")
    print(f"Architecture:     {input_size} -> {HIDDEN_SIZES} -> 1")
    print(f"Instances trained: {model.instances_trained}")
    print(f"Training time:    {train_stats['training_time']:.1f}s")
    print(f"Max memory:       {train_stats['max_memory']:.0f}MB")
    print(f"Test R²:          {eval_results['r2']:.4f}")
    print(f"Test RMSE:        {eval_results['rmse']:.4f}")
    print("\nVariable Importances:")
    for name, val in sorted(imp.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {name}: {val:.6f}")


if __name__ == "__main__":
    main()
