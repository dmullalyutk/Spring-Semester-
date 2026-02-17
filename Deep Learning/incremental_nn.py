"""
Group Assignment 1: Incremental Learning on Large Data

Instruction-aligned implementation:
- Feed-forward neural network with 3 sigmoid hidden layers
- Incremental mini-batch training from CSV chunks
- No test-set use for training or tuning
- Learning curve (instances learned vs moving-average MSE)
- Variable importance (permutation)
- Multiple partial dependence plots
- RAM and training-time reporting
"""

import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil


np.random.seed(42)


FEATURE_COLS = ["sku", "price", "order", "duration", "category"]
TARGET_COL = "quantity"
FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_COLS)}
ROW_COUNT_CACHE = {}


def get_memory_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


class ProgressBar:
    def __init__(self, total, desc="", width=28, min_interval=0.2, unit="it"):
        self.total = int(total) if total is not None else None
        self.desc = desc
        self.width = int(width)
        self.min_interval = float(min_interval)
        self.unit = unit
        self.current = 0
        self.start_time = time.time()
        self.last_render = 0.0
        self.closed = False
        self.last_extra = ""
        self._render(force=True)

    def _format_eta(self, seconds):
        if seconds is None or not np.isfinite(seconds):
            return "--:--"
        seconds = int(max(0, seconds))
        minutes, sec = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{sec:02d}"
        return f"{minutes:02d}:{sec:02d}"

    def _render(self, force=False, extra=""):
        now = time.time()
        if (not force) and (now - self.last_render < self.min_interval):
            return
        self.last_render = now
        self.last_extra = extra
        elapsed = max(now - self.start_time, 1e-9)
        rate = self.current / elapsed
        rate_txt = f"{rate:.1f}{self.unit}/s"

        if self.total is None or self.total <= 0:
            msg = f"\r{self.desc} {self.current}{self.unit} | {rate_txt}"
            if extra:
                msg += f" | {extra}"
            print(msg, end="", flush=True)
            return

        frac = min(max(self.current / self.total, 0.0), 1.0)
        done = int(round(self.width * frac))
        bar = "#" * done + "-" * (self.width - done)
        eta = (self.total - self.current) / rate if rate > 1e-12 else None
        msg = (
            f"\r{self.desc} [{bar}] {100.0 * frac:6.2f}% "
            f"({self.current}/{self.total}) | eta {self._format_eta(eta)} | {rate_txt}"
        )
        if extra:
            msg += f" | {extra}"
        print(msg, end="", flush=True)

    def update(self, n=1, extra=""):
        self.current += int(n)
        self._render(force=False, extra=extra)

    def close(self, extra=""):
        if self.closed:
            return
        self.closed = True
        self._render(force=True, extra=extra or self.last_extra)
        print("", flush=True)


def get_csv_row_count(path):
    cached = ROW_COUNT_CACHE.get(path)
    if cached is not None:
        return cached
    with open(path, "r", encoding="utf-8") as f:
        line_count = sum(1 for _ in f)
    header_rows = 1 if infer_header(path) else 0
    rows = max(0, line_count - header_rows)
    ROW_COUNT_CACHE[path] = rows
    return rows


def estimate_total_chunks(total_rows, chunk_size):
    if chunk_size <= 0:
        return 0
    return int(math.ceil(float(total_rows) / float(chunk_size)))


def estimate_batches_for_chunked_stream(total_rows, chunk_rows, batch_size):
    if total_rows <= 0:
        return 0
    if batch_size <= 0:
        return int(math.ceil(float(total_rows)))
    if chunk_rows <= 0:
        return int(math.ceil(float(total_rows) / float(batch_size)))
    full_chunks = total_rows // chunk_rows
    remainder = total_rows % chunk_rows
    per_full_chunk = int(math.ceil(float(chunk_rows) / float(batch_size)))
    total = full_chunks * per_full_chunk
    if remainder > 0:
        total += int(math.ceil(float(remainder) / float(batch_size)))
    return int(total)


def moving_average(values, window=200):
    if len(values) < window:
        return np.array(values, dtype=np.float64)
    arr = np.asarray(values, dtype=np.float64)
    return np.convolve(arr, np.ones(window) / window, mode="valid")


def compute_r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot <= 1e-12:
        return 0.0
    return 1.0 - (ss_res / ss_tot)


def infer_header(path):
    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    tokens = [t.strip() for t in first_line.split(",")]
    expected = FEATURE_COLS[:]
    expected.insert(2, TARGET_COL)
    return [t.lower() for t in tokens] == expected


def resolve_data_paths():
    if "__file__" in globals():
        script_dir = os.path.dirname(os.path.abspath(__file__))
    else:
        script_dir = os.getcwd()

    def resolve(filename):
        candidates = [
            os.path.join(script_dir, filename),
            os.path.join(os.getcwd(), filename),
            os.path.join(os.getcwd(), "Deep Learning", filename),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return candidates[0]

    train_file = resolve("pricing.csv")
    test_file = resolve("pricing_test.csv")
    output_dir = os.path.dirname(train_file)
    return train_file, test_file, output_dir


def transform_target(y, use_log_target=False):
    if use_log_target:
        return np.log1p(np.maximum(y, 0.0))
    return y


def inverse_transform_target(y, use_log_target=False):
    if use_log_target:
        return np.expm1(y)
    return y


def compute_raw_feature_means(train_file, chunksize=50000):
    n = 0
    sx = np.zeros(len(FEATURE_COLS), dtype=np.float64)
    total_rows = get_csv_row_count(train_file)
    progress = ProgressBar(
        total=estimate_total_chunks(total_rows, chunksize),
        desc="Raw means",
        unit="ch",
    )
    try:
        for chunk in pd.read_csv(train_file, chunksize=chunksize):
            x = chunk[FEATURE_COLS].values.astype(np.float64)
            n += len(chunk)
            sx += np.sum(x, axis=0)
            progress.update(1, extra=f"rows={n}")
    finally:
        progress.close(extra=f"rows={n}")
    if n == 0:
        raise ValueError("Training file has zero rows.")
    return (sx / n).astype(np.float32)


def transform_features(
    x,
    feature_mode="raw",
    sku_hash_buckets=32,
    category_hash_buckets=16,
    add_interactions=False,
):
    xt = np.asarray(x, dtype=np.float64).copy()
    if feature_mode == "log_skew":
        for name in ("price", "order", "duration"):
            idx = FEATURE_INDEX[name]
            xt[:, idx] = np.log1p(np.maximum(xt[:, idx], 0.0))
    if feature_mode in ("hash_sku", "hash_ids"):
        # Treat ids as categorical via hashing, not numeric magnitude.
        n = xt.shape[0]
        sku = np.mod(xt[:, FEATURE_INDEX["sku"]].astype(np.int64), int(sku_hash_buckets))
        category = np.mod(
            xt[:, FEATURE_INDEX["category"]].astype(np.int64), int(category_hash_buckets)
        )

        # Use log transform for skewed continuous columns.
        price = np.log1p(np.maximum(xt[:, FEATURE_INDEX["price"]], 0.0))
        order = np.log1p(np.maximum(xt[:, FEATURE_INDEX["order"]], 0.0))
        duration = np.log1p(np.maximum(xt[:, FEATURE_INDEX["duration"]], 0.0))
        numeric = np.stack([price, order, duration], axis=1)
        if add_interactions:
            interactions = np.stack(
                [price * duration, order * duration, price * order], axis=1
            )
            numeric = np.concatenate([numeric, interactions], axis=1)

        cat_one_hot = np.zeros((n, int(category_hash_buckets)), dtype=np.float64)
        cat_one_hot[np.arange(n), category] = 1.0

        sku_one_hot = np.zeros((n, int(sku_hash_buckets)), dtype=np.float64)
        sku_one_hot[np.arange(n), sku] = 1.0
        return np.concatenate([numeric, cat_one_hot, sku_one_hot], axis=1)
    if add_interactions:
        # Rich second-order interactions for continuous signals.
        price = xt[:, FEATURE_INDEX["price"]]
        order = xt[:, FEATURE_INDEX["order"]]
        duration = xt[:, FEATURE_INDEX["duration"]]
        interactions = np.stack(
            [
                price * order,
                price * duration,
                order * duration,
                price * price,
                order * order,
                duration * duration,
            ],
            axis=1,
        )
        xt = np.concatenate([xt, interactions], axis=1)
    return xt


def compute_stats(
    train_file,
    chunksize=50000,
    use_log_target=False,
    feature_mode="raw",
    sku_hash_buckets=32,
    category_hash_buckets=16,
    add_interactions=False,
):
    n = 0
    sx = None
    sx2 = None
    sy = 0.0
    sy2 = 0.0
    total_rows = get_csv_row_count(train_file)
    progress = ProgressBar(
        total=estimate_total_chunks(total_rows, chunksize),
        desc="Stats",
        unit="ch",
    )
    try:
        for chunk in pd.read_csv(train_file, chunksize=chunksize):
            x_raw = chunk[FEATURE_COLS].values.astype(np.float64)
            x = transform_features(
                x_raw,
                feature_mode=feature_mode,
                sku_hash_buckets=sku_hash_buckets,
                category_hash_buckets=category_hash_buckets,
                add_interactions=add_interactions,
            )
            y = chunk[TARGET_COL].values.astype(np.float64)
            y = transform_target(y, use_log_target=use_log_target)
            if sx is None:
                sx = np.zeros(x.shape[1], dtype=np.float64)
                sx2 = np.zeros(x.shape[1], dtype=np.float64)
            n += len(chunk)
            sx += np.sum(x, axis=0)
            sx2 += np.sum(x * x, axis=0)
            sy += float(np.sum(y))
            sy2 += float(np.sum(y * y))
            progress.update(1, extra=f"rows={n}")
    finally:
        progress.close(extra=f"rows={n}")

    if n == 0:
        raise ValueError("Training file has zero rows.")

    x_mean = sx / n
    x_var = np.maximum((sx2 / n) - (x_mean * x_mean), 1e-12)
    x_std = np.sqrt(x_var)

    y_mean = sy / n
    y_var = max((sy2 / n) - (y_mean * y_mean), 1e-12)
    y_std = np.sqrt(y_var)

    return (
        x_mean.astype(np.float32),
        x_std.astype(np.float32),
        np.float32(y_mean),
        np.float32(y_std),
    )


def std_x(x, mean, std, clip=8.0):
    z = (x - mean) / std
    if clip is not None:
        z = np.clip(z, -clip, clip)
    return z


def std_y(y, mean, std, use_log_target=False):
    yt = transform_target(y, use_log_target=use_log_target)
    return (yt - mean) / std


def unstd_y(y_std, mean, std, use_log_target=False):
    yt = (y_std * std) + mean
    return inverse_transform_target(yt, use_log_target=use_log_target)


def preprocess_x(
    x_raw,
    mean,
    std,
    clip=8.0,
    feature_mode="raw",
    sku_scale=1.0,
    sku_hash_buckets=32,
    category_scale=1.0,
    category_hash_buckets=16,
    add_interactions=False,
):
    xt = transform_features(
        x_raw,
        feature_mode=feature_mode,
        sku_hash_buckets=sku_hash_buckets,
        category_hash_buckets=category_hash_buckets,
        add_interactions=add_interactions,
    )
    z = std_x(xt, mean, std, clip=clip)
    if feature_mode in ("hash_sku", "hash_ids"):
        numeric_count = 3 + (3 if add_interactions else 0)
        cat_start = numeric_count
        cat_end = cat_start + int(category_hash_buckets)
        sku_start = cat_end
        z[:, cat_start:cat_end] = z[:, cat_start:cat_end] * float(category_scale)
        z[:, sku_start:] = z[:, sku_start:] * float(sku_scale)
    else:
        z[:, FEATURE_INDEX["sku"]] = z[:, FEATURE_INDEX["sku"]] * float(sku_scale)
    return z


def split_train_validation_from_file(
    train_file, n_rows=120000, val_ratio=0.2, chunksize=30000, strategy="random"
):
    """
    Build a tuning subset from training file only, then split into train/validation.
    This avoids using pricing_test.csv for hyperparameter tuning.
    """
    parts = []
    kept = 0
    total_rows = get_csv_row_count(train_file)
    target_rows = total_rows if n_rows is None else min(int(n_rows), total_rows)
    progress = ProgressBar(
        total=estimate_total_chunks(target_rows, chunksize),
        desc="Build tune split",
        unit="ch",
    )
    try:
        for chunk in pd.read_csv(train_file, chunksize=chunksize):
            if n_rows is not None and kept >= n_rows:
                break
            if n_rows is None:
                take = len(chunk)
            else:
                need = n_rows - kept
                take = min(need, len(chunk))
            parts.append(chunk.sample(n=take, random_state=42))
            kept += take
            progress.update(1, extra=f"rows={kept}")
    finally:
        progress.close(extra=f"rows={kept}")
    if not parts:
        raise ValueError("Unable to create train/validation split from training file.")

    df = pd.concat(parts, ignore_index=True)

    if strategy == "sku_shift":
        # Validation uses higher-SKU rows to better reflect test distribution shift.
        df_sorted = df.sort_values("sku").reset_index(drop=True)
        cut = int((1.0 - val_ratio) * len(df_sorted))
        train_df = df_sorted.iloc[:cut].reset_index(drop=True)
        val_df = df_sorted.iloc[cut:].reset_index(drop=True)
    else:
        idx = np.random.RandomState(42).permutation(len(df))
        cut = int((1.0 - val_ratio) * len(df))
        train_idx = idx[:cut]
        val_idx = idx[cut:]
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

    return train_df, val_df


class IncrementalNeuralNetwork:
    def __init__(
        self,
        input_size=5,
        hidden_sizes=(128, 64, 32),
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        grad_clip=5.0,
        loss_type="mse",
        huber_delta=1.0,
        weight_decay=0.0,
    ):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.grad_clip = grad_clip
        self.loss_type = loss_type
        self.huber_delta = huber_delta
        self.weight_decay = weight_decay

        sizes = [input_size] + list(hidden_sizes) + [1]
        self.layers = []
        for i in range(len(sizes) - 1):
            lim = np.sqrt(6.0 / (sizes[i] + sizes[i + 1]))
            w = np.random.uniform(-lim, lim, (sizes[i], sizes[i + 1])).astype(np.float32)
            b = np.zeros((1, sizes[i + 1]), dtype=np.float32)
            self.layers.append({"W": w, "b": b})

        self.mw = [np.zeros_like(layer["W"]) for layer in self.layers]
        self.vw = [np.zeros_like(layer["W"]) for layer in self.layers]
        self.mb = [np.zeros_like(layer["b"]) for layer in self.layers]
        self.vb = [np.zeros_like(layer["b"]) for layer in self.layers]
        self.t = 0

        self.mse_history = []
        self.instances_history = []
        self.instances_trained = 0

    @staticmethod
    def sigmoid(x):
        x = np.clip(x, -40, 40)
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def sigmoid_grad(a):
        return a * (1.0 - a)

    def forward(self, x):
        acts = [x]
        cur = x
        for i, layer in enumerate(self.layers):
            z = cur @ layer["W"] + layer["b"]
            if i < len(self.layers) - 1:
                cur = self.sigmoid(z)
            else:
                cur = z
            acts.append(cur)
        return cur, acts

    def train_batch(self, x, y):
        out, acts = self.forward(x)
        y2 = y.reshape(-1, 1)
        err = out - y2
        mse = float(np.mean((out.flatten() - y) ** 2))
        if self.loss_type == "huber":
            abs_err = np.abs(err)
            delta = np.where(abs_err <= self.huber_delta, err, self.huber_delta * np.sign(err))
        else:
            delta = err
        self.t += 1

        for i in range(len(self.layers) - 1, -1, -1):
            w_old = self.layers[i]["W"]
            a_prev = acts[i]

            dW = (a_prev.T @ delta) / x.shape[0]
            db = np.mean(delta, axis=0, keepdims=True)
            if self.weight_decay > 0.0:
                dW = dW + self.weight_decay * w_old

            if self.grad_clip is not None:
                dW = np.clip(dW, -self.grad_clip, self.grad_clip)
                db = np.clip(db, -self.grad_clip, self.grad_clip)

            self.mw[i] = self.beta1 * self.mw[i] + (1 - self.beta1) * dW
            self.vw[i] = self.beta2 * self.vw[i] + (1 - self.beta2) * (dW * dW)
            self.mb[i] = self.beta1 * self.mb[i] + (1 - self.beta1) * db
            self.vb[i] = self.beta2 * self.vb[i] + (1 - self.beta2) * (db * db)

            mw_hat = self.mw[i] / (1 - self.beta1 ** self.t)
            vw_hat = self.vw[i] / (1 - self.beta2 ** self.t)
            mb_hat = self.mb[i] / (1 - self.beta1 ** self.t)
            vb_hat = self.vb[i] / (1 - self.beta2 ** self.t)

            self.layers[i]["W"] = w_old - self.learning_rate * mw_hat / (np.sqrt(vw_hat) + self.eps)
            self.layers[i]["b"] = self.layers[i]["b"] - self.learning_rate * mb_hat / (np.sqrt(vb_hat) + self.eps)

            if i > 0:
                delta = (delta @ w_old.T) * self.sigmoid_grad(acts[i])

        self.instances_trained += x.shape[0]
        self.instances_history.append(self.instances_trained)
        self.mse_history.append(mse)
        return mse

    def predict(self, x):
        cur = x
        for i, layer in enumerate(self.layers):
            z = cur @ layer["W"] + layer["b"]
            if i < len(self.layers) - 1:
                cur = self.sigmoid(z)
            else:
                cur = z
        return cur.flatten()


def train_on_arrays(
    model,
    x,
    y,
    batch_size=128,
    epochs=5,
    lr_decay=1.0,
    min_learning_rate=1e-5,
    show_progress=False,
    progress_desc="Array train",
):
    start = time.time()
    mem = [get_memory_usage_mb()]
    rs = np.random.RandomState(42)
    initial_lr = float(model.learning_rate)
    batches_per_epoch = int(math.ceil(float(x.shape[0]) / float(batch_size)))
    progress = None
    if show_progress:
        progress = ProgressBar(
            total=epochs * batches_per_epoch,
            desc=progress_desc,
            unit="b",
        )

    try:
        for epoch_idx in range(epochs):
            model.learning_rate = max(
                initial_lr * (float(lr_decay) ** epoch_idx),
                float(min_learning_rate),
            )
            perm = rs.permutation(x.shape[0])
            x_ep = x[perm]
            y_ep = y[perm]
            for i in range(0, x_ep.shape[0], batch_size):
                xb = x_ep[i : i + batch_size]
                yb = y_ep[i : i + batch_size]
                mse = model.train_batch(xb, yb)
                if progress is not None:
                    progress.update(
                        1,
                        extra=f"epoch={epoch_idx + 1}/{epochs} lr={model.learning_rate:.6f} mse={mse:.4f}",
                    )
            mem.append(get_memory_usage_mb())
    finally:
        if progress is not None:
            progress.close()

    elapsed = time.time() - start
    return {
        "training_time": elapsed,
        "max_memory": float(np.max(mem)),
        "avg_memory": float(np.mean(mem)),
    }


def evaluate_on_arrays(model, x, y, y_mean, y_std, use_log_target=False):
    pred_std = model.predict(x)
    pred = unstd_y(pred_std, y_mean, y_std, use_log_target=use_log_target)
    mse = float(np.mean((y - pred) ** 2))
    rmse = float(np.sqrt(mse))
    r2 = float(compute_r2(y, pred))
    return {"mse": mse, "rmse": rmse, "r2": r2}


def train_incremental(
    model,
    train_file,
    x_mean,
    x_std,
    y_mean,
    y_std,
    batch_size=128,
    epochs=8,
    chunk_rows=20000,
    x_clip=8.0,
    use_log_target=False,
    feature_mode="raw",
    sku_scale=1.0,
    sku_hash_buckets=32,
    category_scale=1.0,
    category_hash_buckets=16,
    add_interactions=False,
    lr_decay=1.0,
    min_learning_rate=1e-5,
    shuffle_buffer_chunks=1,
):
    print("\n" + "=" * 60)
    print("TRAINING PHASE")
    print("=" * 60)
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Chunk rows: {chunk_rows}")
    print(f"LR decay per epoch: {lr_decay}")
    print(f"Min learning rate: {min_learning_rate}")
    print(f"Shuffle buffer chunks: {shuffle_buffer_chunks}")
    print(f"Initial memory usage: {get_memory_usage_mb():.2f} MB")

    start = time.time()
    mem = [get_memory_usage_mb()]
    log_every_batches = max(1, int(100000 / batch_size))
    initial_lr = float(model.learning_rate)
    total_rows = get_csv_row_count(train_file)
    batches_per_epoch = estimate_batches_for_chunked_stream(
        total_rows=total_rows,
        chunk_rows=chunk_rows,
        batch_size=batch_size,
    )

    for epoch in range(1, epochs + 1):
        model.learning_rate = max(
            initial_lr * (float(lr_decay) ** (epoch - 1)),
            float(min_learning_rate),
        )
        print(f"\nEpoch {epoch}/{epochs} (lr={model.learning_rate:.6f})")
        batch_count = 0
        epoch_bar = ProgressBar(
            total=batches_per_epoch,
            desc=f"Epoch {epoch}/{epochs}",
            unit="b",
        )

        rng = np.random.RandomState(42 + epoch)
        chunk_buffer = []

        def process_chunk(chunk):
            x_chunk = chunk[FEATURE_COLS].values.astype(np.float32)
            y_chunk = chunk[TARGET_COL].values.astype(np.float32)

            x_chunk = preprocess_x(
                x_chunk,
                x_mean,
                x_std,
                clip=x_clip,
                feature_mode=feature_mode,
                sku_scale=sku_scale,
                sku_hash_buckets=sku_hash_buckets,
                category_scale=category_scale,
                category_hash_buckets=category_hash_buckets,
                add_interactions=add_interactions,
            )
            y_chunk = std_y(y_chunk, y_mean, y_std, use_log_target=use_log_target)

            n = x_chunk.shape[0]
            nonlocal batch_count
            for i in range(0, n, batch_size):
                xb = x_chunk[i : i + batch_size]
                yb = y_chunk[i : i + batch_size]
                mse = model.train_batch(xb, yb)
                batch_count += 1
                extra = f"mse={mse:.4f}"

                if batch_count % log_every_batches == 0:
                    m = get_memory_usage_mb()
                    mem.append(m)
                    extra = f"mse={mse:.4f} mem={m:.1f}MB inst={model.instances_trained}"
                epoch_bar.update(1, extra=extra)

        for chunk in pd.read_csv(train_file, chunksize=chunk_rows):
            # Shuffle inside each chunk, then shuffle chunk ordering via a bounded buffer.
            chunk = chunk.sample(frac=1.0, random_state=rng.randint(0, 2**31 - 1)).reset_index(drop=True)
            chunk_buffer.append(chunk)
            if len(chunk_buffer) >= int(max(1, shuffle_buffer_chunks)):
                pick = rng.randint(0, len(chunk_buffer))
                process_chunk(chunk_buffer.pop(pick))

        while chunk_buffer:
            pick = rng.randint(0, len(chunk_buffer))
            process_chunk(chunk_buffer.pop(pick))
        mem.append(get_memory_usage_mb())
        epoch_bar.close(extra=f"inst={model.instances_trained}")

    elapsed = time.time() - start
    return {
        "training_time": elapsed,
        "memory_readings": mem,
        "max_memory": float(np.max(mem)),
        "avg_memory": float(np.mean(mem)),
    }


def load_test_data(test_file):
    has_header = infer_header(test_file)
    if has_header:
        df = pd.read_csv(test_file)
    else:
        df = pd.read_csv(test_file, header=None)
        df.columns = ["sku", "price", "quantity", "order", "duration", "category"]
    x = df[FEATURE_COLS].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)
    return x, y


def align_test_feature_scales(x_test, train_feature_means):
    """
    If a test feature is wildly off-scale vs train (often due to missing fixed divisor),
    rescale by a power of 10 to match train magnitude.
    """
    x = x_test.copy()
    for i, name in enumerate(FEATURE_COLS):
        tr = abs(float(train_feature_means[i])) + 1e-12
        te = abs(float(np.mean(x[:, i]))) + 1e-12
        ratio = te / tr
        if ratio > 50 or ratio < 0.02:
            power = int(np.round(np.log10(ratio)))
            factor = 10.0 ** power
            if factor != 0:
                x[:, i] = x[:, i] / factor
                print(f"Adjusted test feature scale: {name} / {factor:.0e}")
    return x


def evaluate(
    model,
    test_file,
    x_mean,
    x_std,
    y_mean,
    y_std,
    x_clip=8.0,
    use_log_target=False,
    feature_mode="raw",
    sku_scale=1.0,
    sku_hash_buckets=32,
    category_scale=1.0,
    category_hash_buckets=16,
    add_interactions=False,
    train_raw_feature_means=None,
):
    print("\n" + "=" * 60)
    print("EVALUATION PHASE")
    print("=" * 60)
    print("Evaluating on test data...")

    x_test, y_test = load_test_data(test_file)
    if train_raw_feature_means is not None:
        x_test = align_test_feature_scales(x_test, train_raw_feature_means)
    x_test_std = preprocess_x(
        x_test,
        x_mean,
        x_std,
        clip=x_clip,
        feature_mode=feature_mode,
        sku_scale=sku_scale,
        sku_hash_buckets=sku_hash_buckets,
        category_scale=category_scale,
        category_hash_buckets=category_hash_buckets,
        add_interactions=add_interactions,
    )
    y_pred_std = model.predict(x_test_std)
    y_pred = unstd_y(y_pred_std, y_mean, y_std, use_log_target=use_log_target)

    mse = float(np.mean((y_test - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    r2 = float(compute_r2(y_test, y_pred))

    print(f"Test MSE: {mse:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R2: {r2:.4f}")
    return {
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "y_true": y_test,
        "y_pred": y_pred,
    }


def sample_training_rows(train_file, n_rows=5000, chunksize=20000):
    parts = []
    kept = 0
    for chunk in pd.read_csv(train_file, chunksize=chunksize):
        if kept >= n_rows:
            break
        need = n_rows - kept
        take = min(need, len(chunk))
        parts.append(chunk.sample(n=take, random_state=42))
        kept += take
    if not parts:
        raise ValueError("Could not sample training rows.")
    df = pd.concat(parts, ignore_index=True)
    x = df[FEATURE_COLS].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)
    return x, y


def permutation_importance(
    model,
    x_raw_sample,
    y_std_sample,
    x_mean,
    x_std,
    names,
    repeats=5,
    x_clip=8.0,
    feature_mode="raw",
    sku_scale=1.0,
    sku_hash_buckets=32,
    category_scale=1.0,
    category_hash_buckets=16,
    add_interactions=False,
):
    x_std_sample = preprocess_x(
        x_raw_sample,
        x_mean,
        x_std,
        clip=x_clip,
        feature_mode=feature_mode,
        sku_scale=sku_scale,
        sku_hash_buckets=sku_hash_buckets,
        category_scale=category_scale,
        category_hash_buckets=category_hash_buckets,
        add_interactions=add_interactions,
    )
    base = model.predict(x_std_sample)
    base_mse = float(np.mean((base - y_std_sample) ** 2))
    out = {}

    for i, name in enumerate(names):
        deltas = []
        for _ in range(repeats):
            x_raw_perm = x_raw_sample.copy()
            perm = np.random.permutation(x_raw_perm.shape[0])
            x_raw_perm[:, i] = x_raw_perm[perm, i]
            xp = preprocess_x(
                x_raw_perm,
                x_mean,
                x_std,
                clip=x_clip,
                feature_mode=feature_mode,
                sku_scale=sku_scale,
                sku_hash_buckets=sku_hash_buckets,
                category_scale=category_scale,
                category_hash_buckets=category_hash_buckets,
                add_interactions=add_interactions,
            )
            p = model.predict(xp)
            m = float(np.mean((p - y_std_sample) ** 2))
            deltas.append(m - base_mse)
        out[name] = float(np.mean(deltas))
    return out


def partial_dependence_curves(
    model,
    x_raw_sample,
    x_mean,
    x_std,
    y_mean,
    y_std,
    grid_points=40,
    x_clip=8.0,
    use_log_target=False,
    feature_mode="raw",
    sku_scale=1.0,
    sku_hash_buckets=32,
    category_scale=1.0,
    category_hash_buckets=16,
    add_interactions=False,
):
    curves = {}
    for i, name in enumerate(FEATURE_COLS):
        vals = x_raw_sample[:, i]
        grid = np.linspace(np.min(vals), np.max(vals), grid_points)
        yvals = []
        for v in grid:
            x_temp = x_raw_sample.copy()
            x_temp[:, i] = v
            x_temp_std = preprocess_x(
                x_temp,
                x_mean,
                x_std,
                clip=x_clip,
                feature_mode=feature_mode,
                sku_scale=sku_scale,
                sku_hash_buckets=sku_hash_buckets,
                category_scale=category_scale,
                category_hash_buckets=category_hash_buckets,
                add_interactions=add_interactions,
            )
            pred_std = model.predict(x_temp_std)
            pred = unstd_y(pred_std, y_mean, y_std, use_log_target=use_log_target)
            yvals.append(float(np.mean(pred)))
        curves[name] = (grid, np.array(yvals))
    return curves


def plot_learning_curve(instances, mse_history, output_path, window=200):
    ma = moving_average(mse_history, window=window)
    if len(mse_history) >= window:
        x = np.asarray(instances[window - 1 :], dtype=np.int64)
    else:
        x = np.asarray(instances, dtype=np.int64)

    plt.figure(figsize=(12, 6))
    plt.plot(x, ma, linewidth=1.0)
    plt.xlabel("Number of Instances Learned")
    plt.ylabel(f"Moving Average MSE (window={window})")
    plt.title("Learning Curve")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Learning curve saved to {output_path}")


def plot_importance(importances, output_path):
    pairs = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
    names = [p[0] for p in pairs]
    vals = [p[1] for p in pairs]

    plt.figure(figsize=(9, 5))
    plt.barh(range(len(names)), vals)
    plt.yticks(range(len(names)), names)
    plt.gca().invert_yaxis()
    plt.xlabel("Importance (delta MSE on standardized target)")
    plt.title("Permutation Variable Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Variable importance plot saved to {output_path}")


def plot_partial_dependence(curves, output_path):
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    for i, name in enumerate(FEATURE_COLS):
        grid, yvals = curves[name]
        ax = axes[i]
        ax.plot(grid, yvals, linewidth=1.8)
        ax.set_title(name)
        ax.set_xlabel(name)
        ax.set_ylabel("Predicted quantity")
        ax.grid(alpha=0.3)

    for j in range(len(FEATURE_COLS), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Partial Dependence Plots", y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Partial dependence plots saved to {output_path}")


def normalize_val_strategies(val_strategy):
    if isinstance(val_strategy, str):
        return [val_strategy]
    if val_strategy is None:
        return ["random"]
    out = []
    for s in val_strategy:
        if isinstance(s, str) and s and s not in out:
            out.append(s)
    return out if out else ["random"]


def hyperparameter_sweep(
    train_file,
    hidden_sizes,
    sweep_grid,
    label="Sweep",
    val_strategy="random",
    sweep_seeds=(101, 202, 303),
    tuning_rows=120000,
    val_ratio=0.2,
    selection_std_penalty=0.5,
):
    val_strategies = normalize_val_strategies(val_strategy)
    print("\n" + "=" * 60)
    print(f"{label.upper()} (TRAIN-ONLY)")
    print("=" * 60)
    print(
        "Tuning uses validation split(s) from pricing.csv only "
        f"(strategies={val_strategies}, seeds={list(sweep_seeds)}, "
        f"rows={'all' if tuning_rows is None else tuning_rows}, val_ratio={val_ratio})."
    )
    print(
        f"Selection rule: maximize score = mean(val_R2) - {selection_std_penalty:.2f}*std(val_R2)"
    )

    split_data = {}
    for strategy in val_strategies:
        print(f"\nPreparing validation split: {strategy}")
        train_df, val_df = split_train_validation_from_file(
            train_file, n_rows=tuning_rows, val_ratio=val_ratio, strategy=strategy
        )
        split_data[strategy] = {
            "x_train_raw": train_df[FEATURE_COLS].values.astype(np.float32),
            "y_train_raw": train_df[TARGET_COL].values.astype(np.float32),
            "x_val_raw": val_df[FEATURE_COLS].values.astype(np.float32),
            "y_val_raw": val_df[TARGET_COL].values.astype(np.float32),
        }

    results = []
    best = None
    cfg_bar = ProgressBar(total=len(sweep_grid), desc=f"{label} configs", unit="cfg")

    for i, cfg in enumerate(sweep_grid, start=1):
        cfg_hidden = cfg.get("hidden_sizes", hidden_sizes)
        cfg_clip = cfg.get("x_clip", 8.0)
        cfg_log = cfg.get("use_log_target", False)
        cfg_loss = cfg.get("loss_type", "mse")
        cfg_huber_delta = cfg.get("huber_delta", 1.0)
        cfg_weight_decay = cfg.get("weight_decay", 0.0)
        cfg_lr_decay = cfg.get("lr_decay", 1.0)
        cfg_min_lr = cfg.get("min_learning_rate", 1e-5)
        cfg_feature_mode = cfg.get("feature_mode", "raw")
        cfg_sku_scale = cfg.get("sku_scale", 1.0)
        cfg_sku_hash_buckets = cfg.get("sku_hash_buckets", 32)
        cfg_category_scale = cfg.get("category_scale", 1.0)
        cfg_category_hash_buckets = cfg.get("category_hash_buckets", 16)
        cfg_add_interactions = cfg.get("add_interactions", False)

        seed_runs = []
        per_strategy = {}
        for strategy in val_strategies:
            data = split_data[strategy]
            x_train_raw = data["x_train_raw"]
            y_train_raw = data["y_train_raw"]
            x_val_raw = data["x_val_raw"]
            y_val_raw = data["y_val_raw"]

            x_train_tf = transform_features(
                x_train_raw,
                feature_mode=cfg_feature_mode,
                sku_hash_buckets=cfg_sku_hash_buckets,
                category_hash_buckets=cfg_category_hash_buckets,
                add_interactions=cfg_add_interactions,
            )
            x_mean = np.mean(x_train_tf, axis=0).astype(np.float32)
            x_std = np.std(x_train_tf, axis=0).astype(np.float32)
            x_std = np.maximum(x_std, 1e-6)
            y_train_t = transform_target(y_train_raw.astype(np.float64), use_log_target=cfg_log)
            y_mean = np.float32(np.mean(y_train_t))
            y_std = np.float32(max(np.std(y_train_t), 1e-6))

            x_train = preprocess_x(
                x_train_raw,
                x_mean,
                x_std,
                clip=cfg_clip,
                feature_mode=cfg_feature_mode,
                sku_scale=cfg_sku_scale,
                sku_hash_buckets=cfg_sku_hash_buckets,
                category_scale=cfg_category_scale,
                category_hash_buckets=cfg_category_hash_buckets,
                add_interactions=cfg_add_interactions,
            )
            y_train = std_y(y_train_raw, y_mean, y_std, use_log_target=cfg_log)
            x_val = preprocess_x(
                x_val_raw,
                x_mean,
                x_std,
                clip=cfg_clip,
                feature_mode=cfg_feature_mode,
                sku_scale=cfg_sku_scale,
                sku_hash_buckets=cfg_sku_hash_buckets,
                category_scale=cfg_category_scale,
                category_hash_buckets=cfg_category_hash_buckets,
                add_interactions=cfg_add_interactions,
            )

            for seed in sweep_seeds:
                np.random.seed(seed)
                model = IncrementalNeuralNetwork(
                    input_size=x_train.shape[1],
                    hidden_sizes=cfg_hidden,
                    learning_rate=cfg["learning_rate"],
                    loss_type=cfg_loss,
                    huber_delta=cfg_huber_delta,
                    weight_decay=cfg_weight_decay,
                )
                stats = train_on_arrays(
                    model=model,
                    x=x_train,
                    y=y_train,
                    batch_size=cfg["batch_size"],
                    epochs=cfg["epochs"],
                    lr_decay=cfg_lr_decay,
                    min_learning_rate=cfg_min_lr,
                    show_progress=True,
                    progress_desc=f"{label} cfg {i}/{len(sweep_grid)} {strategy} seed {seed}",
                )
                val_metrics = evaluate_on_arrays(
                    model, x_val, y_val_raw, y_mean, y_std, use_log_target=cfg_log
                )
                seed_runs.append(
                    {
                        "strategy": strategy,
                        "seed": seed,
                        "val_r2": val_metrics["r2"],
                        "val_rmse": val_metrics["rmse"],
                        "time": stats["training_time"],
                        "max_memory": stats["max_memory"],
                    }
                )

            strat_runs = [r for r in seed_runs if r["strategy"] == strategy]
            per_strategy[strategy] = {
                "val_r2": float(np.mean([r["val_r2"] for r in strat_runs])),
                "val_r2_std": float(np.std([r["val_r2"] for r in strat_runs])),
                "val_rmse": float(np.mean([r["val_rmse"] for r in strat_runs])),
                "val_rmse_std": float(np.std([r["val_rmse"] for r in strat_runs])),
            }

        r2_vals = [r["val_r2"] for r in seed_runs]
        rmse_vals = [r["val_rmse"] for r in seed_runs]
        r2_mean = float(np.mean(r2_vals))
        r2_std = float(np.std(r2_vals))
        rmse_mean = float(np.mean(rmse_vals))
        rmse_std = float(np.std(rmse_vals))
        selection_score = float(r2_mean - (selection_std_penalty * r2_std))

        row = {
            "config": cfg,
            "val_r2": r2_mean,
            "val_r2_std": r2_std,
            "val_rmse": rmse_mean,
            "val_rmse_std": rmse_std,
            "selection_score": selection_score,
            "time": float(np.mean([r["time"] for r in seed_runs])),
            "max_memory": float(np.mean([r["max_memory"] for r in seed_runs])),
            "seed_runs": seed_runs,
            "per_strategy": per_strategy,
        }
        results.append(row)

        print(
            f"[{i}/{len(sweep_grid)}] batch={cfg['batch_size']} epochs={cfg['epochs']} "
            f"lr={cfg['learning_rate']:.5f} log_target={cfg_log} clip={cfg_clip} "
            f"decay={cfg_lr_decay:.4f} min_lr={cfg_min_lr:.6f} "
            f"hidden={cfg_hidden} loss={cfg_loss} wd={cfg_weight_decay} "
            f"feat={cfg_feature_mode} sku_scale={cfg_sku_scale} "
            f"cat_scale={cfg_category_scale} hash_buckets(sku/cat)="
            f"{cfg_sku_hash_buckets}/{cfg_category_hash_buckets} "
            f"interact={cfg_add_interactions} "
            f"-> score={row['selection_score']:.4f}, "
            f"val_R2={row['val_r2']:.4f}+/-{row['val_r2_std']:.4f}, "
            f"val_RMSE={row['val_rmse']:.4f}+/-{row['val_rmse_std']:.4f}, "
            f"time={row['time']:.2f}s"
        )
        split_txt = "; ".join(
            [
                f"{name}:R2={vals['val_r2']:.4f}+/-{vals['val_r2_std']:.4f}"
                for name, vals in per_strategy.items()
            ]
        )
        print(f"    split_scores -> {split_txt}")

        is_better = False
        if best is None:
            is_better = True
        elif row["selection_score"] > best["selection_score"] + 1e-12:
            is_better = True
        elif (
            abs(row["selection_score"] - best["selection_score"]) <= 1e-12
            and row["val_r2"] > best["val_r2"]
        ):
            is_better = True
        if is_better:
            best = row

        cfg_bar.update(
            1,
            extra=(
                f"best_score={best['selection_score']:.4f} "
                f"best_R2={best['val_r2']:.4f}"
                if best is not None
                else ""
            ),
        )

    cfg_bar.close(
        extra=(
            f"best_score={best['selection_score']:.4f} best_R2={best['val_r2']:.4f}"
            if best is not None
            else ""
        )
    )

    print("\nBest config from sweep:")
    print(
        f"  batch={best['config']['batch_size']}, epochs={best['config']['epochs']}, "
        f"lr={best['config']['learning_rate']:.5f}, "
        f"log_target={best['config'].get('use_log_target', False)}, "
        f"clip={best['config'].get('x_clip', 8.0)}, "
        f"decay={best['config'].get('lr_decay', 1.0):.4f}, "
        f"min_lr={best['config'].get('min_learning_rate', 1e-5):.6f}, "
        f"hidden={best['config'].get('hidden_sizes', hidden_sizes)}, "
        f"loss={best['config'].get('loss_type', 'mse')}, "
        f"wd={best['config'].get('weight_decay', 0.0)}, "
        f"feat={best['config'].get('feature_mode', 'raw')}, "
        f"sku_scale={best['config'].get('sku_scale', 1.0)}, "
        f"cat_scale={best['config'].get('category_scale', 1.0)}, "
        f"hash_buckets={best['config'].get('sku_hash_buckets', 32)}/"
        f"{best['config'].get('category_hash_buckets', 16)}, "
        f"interact={best['config'].get('add_interactions', False)}, "
        f"score={best['selection_score']:.4f}, "
        f"val_R2={best['val_r2']:.4f}+/-{best['val_r2_std']:.4f}"
    )
    return best, results


def write_experiment_changelog(
    path,
    sweep_sections,
    best_config,
    final_results,
    train_stats,
    val_strategy="random",
    selection_std_penalty=0.5,
):
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    val_strategies = normalize_val_strategies(val_strategy)
    lines = []
    lines.append(f"## {now}")
    lines.append("")
    lines.append(f"Validation Strategy: {val_strategies}")
    lines.append(
        f"Selection Rule: mean(val_R2) - {selection_std_penalty:.2f}*std(val_R2)"
    )
    lines.append("")
    lines.append("### Sweep Results (train/validation on pricing.csv only)")
    for section in sweep_sections:
        lines.append(f"- {section['label']}:")
        for row in section["results"]:
            cfg = row["config"]
            lines.append(
                "  "
                f"batch={cfg['batch_size']}, epochs={cfg['epochs']}, lr={cfg['learning_rate']:.5f}, "
                f"log_target={cfg.get('use_log_target', False)}, clip={cfg.get('x_clip', 8.0)}, "
                f"decay={cfg.get('lr_decay', 1.0):.4f}, min_lr={cfg.get('min_learning_rate', 1e-5):.6f}, "
                f"hidden={cfg.get('hidden_sizes', (128, 64, 32))}, "
                f"loss={cfg.get('loss_type', 'mse')}, wd={cfg.get('weight_decay', 0.0)}, "
                f"feat={cfg.get('feature_mode', 'raw')}, sku_scale={cfg.get('sku_scale', 1.0)}, "
                f"cat_scale={cfg.get('category_scale', 1.0)}, "
                f"hash_buckets={cfg.get('sku_hash_buckets', 32)}/{cfg.get('category_hash_buckets', 16)}, "
                f"interact={cfg.get('add_interactions', False)} "
                f"=> score={row.get('selection_score', row['val_r2']):.4f}, "
                f"val_R2={row['val_r2']:.4f}+/-{row.get('val_r2_std', 0.0):.4f}, "
                f"val_RMSE={row['val_rmse']:.4f}+/-{row.get('val_rmse_std', 0.0):.4f}, "
                f"time={row['time']:.2f}s, max_mem={row['max_memory']:.2f}MB"
            )
            if "per_strategy" in row:
                split_txt = "; ".join(
                    [
                        f"{name}:R2={vals['val_r2']:.4f}+/-{vals['val_r2_std']:.4f}"
                        for name, vals in row["per_strategy"].items()
                    ]
                )
                lines.append("  " + f"splits: {split_txt}")
    lines.append("")
    lines.append("### Selected Config")
    lines.append(
        "- "
        f"batch={best_config['batch_size']}, epochs={best_config['epochs']}, "
        f"lr={best_config['learning_rate']:.5f}, log_target={best_config.get('use_log_target', False)}, "
        f"clip={best_config.get('x_clip', 8.0)}, "
        f"decay={best_config.get('lr_decay', 1.0):.4f}, "
        f"min_lr={best_config.get('min_learning_rate', 1e-5):.6f}, "
        f"hidden={best_config.get('hidden_sizes', (128, 64, 32))}, "
        f"loss={best_config.get('loss_type', 'mse')}, wd={best_config.get('weight_decay', 0.0)}, "
        f"feat={best_config.get('feature_mode', 'raw')}, sku_scale={best_config.get('sku_scale', 1.0)}, "
        f"cat_scale={best_config.get('category_scale', 1.0)}, "
        f"score={best_config.get('selection_score', float('nan')):.4f}, "
        f"hash_buckets={best_config.get('sku_hash_buckets', 32)}/"
        f"{best_config.get('category_hash_buckets', 16)}, "
        f"interact={best_config.get('add_interactions', False)}"
    )
    lines.append("")
    lines.append("### Final Test Result (pricing_test.csv)")
    lines.append(
        "- "
        f"R2={final_results['r2']:.4f}, RMSE={final_results['rmse']:.4f}, "
        f"TrainTime={train_stats['training_time']:.2f}s, MaxMem={train_stats['max_memory']:.2f}MB"
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    print("=" * 60)
    print("Group Assignment 1: Incremental Learning on Large Data")
    print("=" * 60)

    train_file, test_file, output_dir = resolve_data_paths()
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")

    # Tunable training configuration.
    batch_size = 64
    epochs = 18
    chunk_rows = 50000
    hidden_sizes = (192, 96, 48)
    learning_rate = 8e-4
    x_clip = 9.0
    use_log_target = False
    loss_type = "mse"
    huber_delta = 1.0
    weight_decay = 0.0
    lr_decay = 0.992
    min_learning_rate = 1e-5
    feature_mode = "log_skew"
    sku_scale = 0.5
    sku_hash_buckets = 32
    category_scale = 1.0
    category_hash_buckets = 16
    add_interactions = True
    shuffle_buffer_chunks = 16

    # Sweep is train-only (uses a validation split from pricing.csv, not pricing_test.csv).
    run_sweep = True
    val_strategy = ("sku_shift", "random")
    selection_std_penalty = 0.5
    sweep_seeds = (101, 202, 303)
    tuning_rows = None
    tuning_val_ratio = 0.2
    sweep_grid = [
        # Baseline + tiny regularization sweep.
        {"batch_size": 48, "epochs": 22, "learning_rate": 0.0007, "use_log_target": False, "x_clip": 9.0, "hidden_sizes": (192, 96, 48), "loss_type": "mse", "weight_decay": 0.0, "feature_mode": "log_skew", "sku_scale": 0.5, "category_scale": 1.0, "add_interactions": True, "lr_decay": 0.993, "min_learning_rate": 1e-5},
        {"batch_size": 48, "epochs": 22, "learning_rate": 0.0007, "use_log_target": False, "x_clip": 9.0, "hidden_sizes": (192, 96, 48), "loss_type": "mse", "weight_decay": 1e-6, "feature_mode": "log_skew", "sku_scale": 0.5, "category_scale": 1.0, "add_interactions": True, "lr_decay": 0.993, "min_learning_rate": 1e-5},
        {"batch_size": 48, "epochs": 22, "learning_rate": 0.0007, "use_log_target": False, "x_clip": 9.0, "hidden_sizes": (192, 96, 48), "loss_type": "mse", "weight_decay": 3e-6, "feature_mode": "log_skew", "sku_scale": 0.5, "category_scale": 1.0, "add_interactions": True, "lr_decay": 0.993, "min_learning_rate": 1e-5},
        {"batch_size": 48, "epochs": 22, "learning_rate": 0.0007, "use_log_target": False, "x_clip": 9.0, "hidden_sizes": (192, 96, 48), "loss_type": "mse", "weight_decay": 1e-5, "feature_mode": "log_skew", "sku_scale": 0.5, "category_scale": 1.0, "add_interactions": True, "lr_decay": 0.993, "min_learning_rate": 1e-5},
        # Capacity sweep with same schedule.
        {"batch_size": 48, "epochs": 22, "learning_rate": 0.0007, "use_log_target": False, "x_clip": 9.0, "hidden_sizes": (160, 80, 40), "loss_type": "mse", "weight_decay": 1e-6, "feature_mode": "log_skew", "sku_scale": 0.5, "category_scale": 1.0, "add_interactions": True, "lr_decay": 0.993, "min_learning_rate": 1e-5},
        {"batch_size": 48, "epochs": 22, "learning_rate": 0.0007, "use_log_target": False, "x_clip": 9.0, "hidden_sizes": (128, 64, 32), "loss_type": "mse", "weight_decay": 1e-6, "feature_mode": "log_skew", "sku_scale": 0.5, "category_scale": 1.0, "add_interactions": True, "lr_decay": 0.993, "min_learning_rate": 1e-5},
    ]
    focused_grid = [
        # Epoch-window tuning around strongest setup.
        {"batch_size": 48, "epochs": 18, "learning_rate": 0.0007, "use_log_target": False, "x_clip": 9.0, "hidden_sizes": (192, 96, 48), "loss_type": "mse", "weight_decay": 1e-6, "feature_mode": "log_skew", "sku_scale": 0.5, "category_scale": 1.0, "add_interactions": True, "lr_decay": 0.993, "min_learning_rate": 1e-5},
        {"batch_size": 48, "epochs": 22, "learning_rate": 0.0007, "use_log_target": False, "x_clip": 9.0, "hidden_sizes": (192, 96, 48), "loss_type": "mse", "weight_decay": 1e-6, "feature_mode": "log_skew", "sku_scale": 0.5, "category_scale": 1.0, "add_interactions": True, "lr_decay": 0.993, "min_learning_rate": 1e-5},
        {"batch_size": 48, "epochs": 26, "learning_rate": 0.0007, "use_log_target": False, "x_clip": 9.0, "hidden_sizes": (192, 96, 48), "loss_type": "mse", "weight_decay": 1e-6, "feature_mode": "log_skew", "sku_scale": 0.5, "category_scale": 1.0, "add_interactions": True, "lr_decay": 0.993, "min_learning_rate": 1e-5},
        {"batch_size": 48, "epochs": 18, "learning_rate": 0.0007, "use_log_target": False, "x_clip": 9.0, "hidden_sizes": (192, 96, 48), "loss_type": "mse", "weight_decay": 3e-6, "feature_mode": "log_skew", "sku_scale": 0.5, "category_scale": 1.0, "add_interactions": True, "lr_decay": 0.993, "min_learning_rate": 1e-5},
        {"batch_size": 48, "epochs": 22, "learning_rate": 0.0007, "use_log_target": False, "x_clip": 9.0, "hidden_sizes": (192, 96, 48), "loss_type": "mse", "weight_decay": 3e-6, "feature_mode": "log_skew", "sku_scale": 0.5, "category_scale": 1.0, "add_interactions": True, "lr_decay": 0.993, "min_learning_rate": 1e-5},
        {"batch_size": 48, "epochs": 26, "learning_rate": 0.0007, "use_log_target": False, "x_clip": 9.0, "hidden_sizes": (192, 96, 48), "loss_type": "mse", "weight_decay": 3e-6, "feature_mode": "log_skew", "sku_scale": 0.5, "category_scale": 1.0, "add_interactions": True, "lr_decay": 0.993, "min_learning_rate": 1e-5},
    ]
    sweep_sections = []
    selection_score = float("nan")

    if run_sweep:
        coarse_best, coarse_results = hyperparameter_sweep(
            train_file=train_file,
            hidden_sizes=hidden_sizes,
            sweep_grid=sweep_grid,
            label="Coarse Sweep",
            val_strategy=val_strategy,
            sweep_seeds=sweep_seeds,
            tuning_rows=tuning_rows,
            val_ratio=tuning_val_ratio,
            selection_std_penalty=selection_std_penalty,
        )
        sweep_sections.append({"label": "Coarse Sweep", "results": coarse_results})

        focused_best, focused_results = hyperparameter_sweep(
            train_file=train_file,
            hidden_sizes=coarse_best["config"].get("hidden_sizes", hidden_sizes),
            sweep_grid=focused_grid,
            label="Focused Sweep",
            val_strategy=val_strategy,
            sweep_seeds=sweep_seeds,
            tuning_rows=tuning_rows,
            val_ratio=tuning_val_ratio,
            selection_std_penalty=selection_std_penalty,
        )
        sweep_sections.append({"label": "Focused Sweep", "results": focused_results})

        best = (
            focused_best
            if focused_best["selection_score"] >= coarse_best["selection_score"]
            else coarse_best
        )
        batch_size = best["config"]["batch_size"]
        epochs = best["config"]["epochs"]
        learning_rate = best["config"]["learning_rate"]
        x_clip = best["config"].get("x_clip", 8.0)
        use_log_target = best["config"].get("use_log_target", False)
        hidden_sizes = best["config"].get("hidden_sizes", hidden_sizes)
        loss_type = best["config"].get("loss_type", "mse")
        huber_delta = best["config"].get("huber_delta", 1.0)
        weight_decay = best["config"].get("weight_decay", 0.0)
        lr_decay = best["config"].get("lr_decay", 1.0)
        min_learning_rate = best["config"].get("min_learning_rate", 1e-5)
        feature_mode = best["config"].get("feature_mode", "raw")
        sku_scale = best["config"].get("sku_scale", 1.0)
        sku_hash_buckets = best["config"].get("sku_hash_buckets", 32)
        category_scale = best["config"].get("category_scale", 1.0)
        category_hash_buckets = best["config"].get("category_hash_buckets", 16)
        add_interactions = best["config"].get("add_interactions", False)
        selection_score = best.get("selection_score", best.get("val_r2", float("nan")))
        print(
            f"\nUsing best tuned config for full training: "
            f"batch={batch_size}, epochs={epochs}, lr={learning_rate:.5f}, "
            f"decay={lr_decay:.4f}, min_lr={min_learning_rate:.6f}, "
            f"log_target={use_log_target}, clip={x_clip}, hidden={hidden_sizes}, "
            f"loss={loss_type}, wd={weight_decay}, feat={feature_mode}, "
            f"sku_scale={sku_scale}, cat_scale={category_scale}, "
            f"hash_buckets={sku_hash_buckets}/{category_hash_buckets}, "
            f"interact={add_interactions}, score={selection_score:.4f}"
        )

    print(f"\nInitial memory usage: {get_memory_usage_mb():.2f} MB")
    print("Computing train-set feature/target statistics...")
    x_mean, x_std, y_mean, y_std = compute_stats(
        train_file,
        use_log_target=use_log_target,
        feature_mode=feature_mode,
        sku_hash_buckets=sku_hash_buckets,
        category_hash_buckets=category_hash_buckets,
        add_interactions=add_interactions,
    )
    train_raw_feature_means = compute_raw_feature_means(train_file)
    print("Feature means:", np.round(x_mean, 4))
    print("Feature stds: ", np.round(x_std, 4))
    print(f"Target mean/std: {y_mean:.4f} / {y_std:.4f}")

    print("\nInitializing neural network...")
    print(f"Architecture: {len(x_mean)} -> {hidden_sizes} -> 1")
    print("Activation: Sigmoid (hidden), Linear (output)")

    model = IncrementalNeuralNetwork(
        input_size=len(x_mean),
        hidden_sizes=hidden_sizes,
        learning_rate=learning_rate,
        loss_type=loss_type,
        huber_delta=huber_delta,
        weight_decay=weight_decay,
    )

    train_stats = train_incremental(
        model=model,
        train_file=train_file,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        batch_size=batch_size,
        epochs=epochs,
        chunk_rows=chunk_rows,
        x_clip=x_clip,
        use_log_target=use_log_target,
        feature_mode=feature_mode,
        sku_scale=sku_scale,
        sku_hash_buckets=sku_hash_buckets,
        category_scale=category_scale,
        category_hash_buckets=category_hash_buckets,
        add_interactions=add_interactions,
        lr_decay=lr_decay,
        min_learning_rate=min_learning_rate,
        shuffle_buffer_chunks=shuffle_buffer_chunks,
    )

    print("\nTraining completed.")
    print(f"Training time: {train_stats['training_time']:.2f} seconds")
    print(f"Max memory usage: {train_stats['max_memory']:.2f} MB")
    print(f"Average memory usage: {train_stats['avg_memory']:.2f} MB")

    eval_results = evaluate(
        model,
        test_file,
        x_mean,
        x_std,
        y_mean,
        y_std,
        x_clip=x_clip,
        use_log_target=use_log_target,
        feature_mode=feature_mode,
        sku_scale=sku_scale,
        sku_hash_buckets=sku_hash_buckets,
        category_scale=category_scale,
        category_hash_buckets=category_hash_buckets,
        add_interactions=add_interactions,
        train_raw_feature_means=train_raw_feature_means,
    )

    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    learning_curve_path = os.path.join(output_dir, "learning_curve.png")
    importance_path = os.path.join(output_dir, "variable_importance.png")
    pdp_path = os.path.join(output_dir, "partial_dependence.png")

    plot_learning_curve(
        instances=model.instances_history,
        mse_history=model.mse_history,
        output_path=learning_curve_path,
        window=200,
    )

    # Use training sample for interpretation to avoid any test leakage.
    x_raw_sample, y_raw_sample = sample_training_rows(train_file, n_rows=5000)
    y_std_sample = std_y(y_raw_sample, y_mean, y_std, use_log_target=use_log_target)

    print("\nCalculating variable importance...")
    importances = permutation_importance(
        model=model,
        x_raw_sample=x_raw_sample,
        y_std_sample=y_std_sample,
        x_mean=x_mean,
        x_std=x_std,
        names=FEATURE_COLS,
        repeats=5,
        x_clip=x_clip,
        feature_mode=feature_mode,
        sku_scale=sku_scale,
        sku_hash_buckets=sku_hash_buckets,
        category_scale=category_scale,
        category_hash_buckets=category_hash_buckets,
        add_interactions=add_interactions,
    )
    plot_importance(importances, importance_path)

    print("\nGenerating partial dependence plots...")
    curves = partial_dependence_curves(
        model=model,
        x_raw_sample=x_raw_sample,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=y_mean,
        y_std=y_std,
        grid_points=40,
        x_clip=x_clip,
        use_log_target=use_log_target,
        feature_mode=feature_mode,
        sku_scale=sku_scale,
        sku_hash_buckets=sku_hash_buckets,
        category_scale=category_scale,
        category_hash_buckets=category_hash_buckets,
        add_interactions=add_interactions,
    )
    plot_partial_dependence(curves, pdp_path)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Model Architecture: {len(x_mean)} -> {hidden_sizes} -> 1")
    print(f"Batch Size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning Rate: {learning_rate}")
    print(f"LR Decay: {lr_decay}")
    print(f"Min Learning Rate: {min_learning_rate}")
    print(f"Use Log Target: {use_log_target}")
    print(f"Input Clip: {x_clip}")
    print(f"Chunk Rows: {chunk_rows}")
    print(f"Shuffle Buffer Chunks: {shuffle_buffer_chunks}")
    print(f"Loss Type: {loss_type}")
    print(f"Weight Decay: {weight_decay}")
    print(f"Feature Mode: {feature_mode}")
    print(f"SKU Scale: {sku_scale}")
    print(f"Category Scale: {category_scale}")
    print(f"SKU Hash Buckets: {sku_hash_buckets}")
    print(f"Category Hash Buckets: {category_hash_buckets}")
    print(f"Use Interactions: {add_interactions}")
    if run_sweep:
        print(f"Selection Score (mean-penalty*std): {selection_score:.4f}")
    print(f"Total Instances Trained: {model.instances_trained}")
    print(f"Training Time: {train_stats['training_time']:.2f} seconds")
    print(f"Max Memory Usage: {train_stats['max_memory']:.2f} MB")
    print(f"Test R2: {eval_results['r2']:.4f}")
    print(f"Test RMSE: {eval_results['rmse']:.4f}")

    print("\nVariable Importances:")
    for name, val in sorted(importances.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {name}: {val:.8f} ({val:.3e})")

    print("\nOutput files generated:")
    print(f"  - {learning_curve_path}")
    print(f"  - {importance_path}")
    print(f"  - {pdp_path}")

    changelog_path = os.path.join(output_dir, "EXPERIMENT_CHANGELOG.md")
    write_experiment_changelog(
        path=changelog_path,
        sweep_sections=sweep_sections if run_sweep else [],
        best_config={
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "use_log_target": use_log_target,
            "x_clip": x_clip,
            "hidden_sizes": hidden_sizes,
            "loss_type": loss_type,
            "huber_delta": huber_delta,
            "weight_decay": weight_decay,
            "lr_decay": lr_decay,
            "min_learning_rate": min_learning_rate,
            "feature_mode": feature_mode,
            "sku_scale": sku_scale,
            "sku_hash_buckets": sku_hash_buckets,
            "category_scale": category_scale,
            "category_hash_buckets": category_hash_buckets,
            "add_interactions": add_interactions,
            "selection_score": selection_score,
        },
        final_results=eval_results,
        train_stats=train_stats,
        val_strategy=val_strategy,
        selection_std_penalty=selection_std_penalty,
    )
    print(f"  - {changelog_path}")

    return model, eval_results, train_stats


if __name__ == "__main__":
    main()
