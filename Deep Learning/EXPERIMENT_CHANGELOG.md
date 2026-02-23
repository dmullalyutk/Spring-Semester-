## 2026-02-13 12:51:11

### Sweep Results (train/validation on pricing.csv only)
- batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32) => val_R2=0.5297, val_RMSE=37.9700, time=6.08s, max_mem=115.77MB
- batch=64, epochs=10, lr=0.00070, log_target=True, clip=8.0, hidden=(128, 64, 32) => val_R2=0.4830, val_RMSE=39.8094, time=6.05s, max_mem=129.73MB
- batch=64, epochs=12, lr=0.00070, log_target=True, clip=10.0, hidden=(128, 64, 32) => val_R2=0.4183, val_RMSE=42.2283, time=7.27s, max_mem=129.45MB
- batch=96, epochs=10, lr=0.00050, log_target=True, clip=10.0, hidden=(160, 96, 48) => val_R2=0.4742, val_RMSE=40.1474, time=5.88s, max_mem=129.76MB
- batch=64, epochs=12, lr=0.00100, log_target=True, clip=12.0, hidden=(128, 64, 32) => val_R2=0.4319, val_RMSE=41.7312, time=6.89s, max_mem=133.80MB
- batch=128, epochs=10, lr=0.00070, log_target=False, clip=10.0, hidden=(160, 96, 48) => val_R2=0.5097, val_RMSE=38.7683, time=5.44s, max_mem=133.87MB

### Selected Config
- batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32)

### Final Test Result (pricing_test.csv)
- R2=0.4701, RMSE=36.8869, TrainTime=33.92s, MaxMem=127.15MB

---
## 2026-02-13 12:55:21

### Sweep Results (train/validation on pricing.csv only)
- Coarse Sweep:
  batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32) => val_R2=0.5297, val_RMSE=37.9700, time=5.82s, max_mem=115.91MB
  batch=64, epochs=10, lr=0.00070, log_target=True, clip=8.0, hidden=(128, 64, 32) => val_R2=0.4830, val_RMSE=39.8094, time=5.75s, max_mem=129.68MB
  batch=64, epochs=12, lr=0.00070, log_target=True, clip=10.0, hidden=(128, 64, 32) => val_R2=0.4183, val_RMSE=42.2283, time=6.90s, max_mem=130.33MB
  batch=96, epochs=10, lr=0.00050, log_target=True, clip=10.0, hidden=(160, 96, 48) => val_R2=0.4742, val_RMSE=40.1474, time=5.99s, max_mem=130.41MB
  batch=64, epochs=12, lr=0.00100, log_target=True, clip=12.0, hidden=(128, 64, 32) => val_R2=0.4319, val_RMSE=41.7312, time=6.95s, max_mem=135.18MB
  batch=128, epochs=10, lr=0.00070, log_target=False, clip=10.0, hidden=(160, 96, 48) => val_R2=0.5097, val_RMSE=38.7683, time=5.50s, max_mem=135.25MB
- Focused Sweep:
  batch=48, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32) => val_R2=0.5378, val_RMSE=37.6431, time=5.62s, max_mem=133.16MB
  batch=64, epochs=8, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32) => val_R2=0.5074, val_RMSE=38.8586, time=4.76s, max_mem=134.08MB
  batch=64, epochs=12, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32) => val_R2=0.5309, val_RMSE=37.9233, time=7.17s, max_mem=133.80MB
  batch=64, epochs=10, lr=0.00050, log_target=False, clip=8.0, hidden=(128, 64, 32) => val_R2=0.5136, val_RMSE=38.6141, time=6.14s, max_mem=134.08MB
  batch=64, epochs=10, lr=0.00090, log_target=False, clip=8.0, hidden=(128, 64, 32) => val_R2=0.5094, val_RMSE=38.7831, time=5.84s, max_mem=134.02MB
  batch=96, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32) => val_R2=0.4991, val_RMSE=39.1878, time=4.35s, max_mem=134.20MB

### Selected Config
- batch=48, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32)

### Final Test Result (pricing_test.csv)
- R2=0.4531, RMSE=37.4732, TrainTime=33.40s, MaxMem=130.36MB

---
## 2026-02-13 12:59:03

### Sweep Results (train/validation on pricing.csv only)
- Coarse Sweep:
  batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0 => val_R2=0.5297, val_RMSE=37.9700, time=5.96s, max_mem=116.56MB
  batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=huber, wd=0.0 => val_R2=0.5109, val_RMSE=38.7234, time=6.01s, max_mem=129.16MB
  batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=huber, wd=0.0 => val_R2=0.5109, val_RMSE=38.7239, time=6.14s, max_mem=130.10MB
  batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=1e-05 => val_R2=0.5201, val_RMSE=38.3553, time=6.31s, max_mem=130.12MB
  batch=96, epochs=10, lr=0.00050, log_target=False, clip=10.0, hidden=(160, 96, 48), loss=mse, wd=1e-05 => val_R2=0.5017, val_RMSE=39.0861, time=6.09s, max_mem=130.25MB
  batch=128, epochs=10, lr=0.00070, log_target=False, clip=10.0, hidden=(160, 96, 48), loss=huber, wd=1e-05 => val_R2=0.4978, val_RMSE=39.2357, time=5.59s, max_mem=134.04MB
- Focused Sweep:
  batch=48, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0 => val_R2=0.5378, val_RMSE=37.6431, time=5.55s, max_mem=133.19MB
  batch=64, epochs=8, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0 => val_R2=0.5074, val_RMSE=38.8586, time=4.78s, max_mem=134.57MB
  batch=64, epochs=12, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0 => val_R2=0.5309, val_RMSE=37.9233, time=7.29s, max_mem=134.57MB
  batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=huber, wd=0.0 => val_R2=0.5136, val_RMSE=38.6154, time=6.15s, max_mem=134.57MB
  batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=huber, wd=1e-05 => val_R2=0.5032, val_RMSE=39.0250, time=6.19s, max_mem=134.51MB
  batch=64, epochs=10, lr=0.00050, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=1e-05 => val_R2=0.5052, val_RMSE=38.9479, time=6.16s, max_mem=134.51MB

### Selected Config
- batch=48, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0

### Final Test Result (pricing_test.csv)
- R2=0.4531, RMSE=37.4732, TrainTime=32.51s, MaxMem=130.41MB

---
## 2026-02-13 13:03:08

### Sweep Results (train/validation on pricing.csv only)
- Coarse Sweep:
  batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=raw => val_R2=0.5297, val_RMSE=37.9700, time=8.06s, max_mem=125.88MB
  batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=log_skew => val_R2=0.5431, val_RMSE=37.4241, time=7.76s, max_mem=151.00MB
  batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=huber, wd=0.0, feat=raw => val_R2=0.5147, val_RMSE=38.5714, time=7.82s, max_mem=149.56MB
  batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=1e-05, feat=raw => val_R2=0.5201, val_RMSE=38.3553, time=7.96s, max_mem=150.15MB
  batch=96, epochs=10, lr=0.00050, log_target=False, clip=10.0, hidden=(160, 96, 48), loss=mse, wd=1e-05, feat=raw => val_R2=0.5017, val_RMSE=39.0861, time=8.52s, max_mem=151.63MB
  batch=128, epochs=10, lr=0.00070, log_target=False, clip=10.0, hidden=(160, 96, 48), loss=huber, wd=1e-05, feat=raw => val_R2=0.4978, val_RMSE=39.2357, time=7.88s, max_mem=152.14MB
- Focused Sweep:
  batch=48, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=raw => val_R2=0.5378, val_RMSE=37.6431, time=7.44s, max_mem=151.70MB
  batch=64, epochs=8, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=raw => val_R2=0.5074, val_RMSE=38.8586, time=6.09s, max_mem=151.21MB
  batch=64, epochs=12, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=raw => val_R2=0.5309, val_RMSE=37.9233, time=9.16s, max_mem=154.12MB
  batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=log_skew => val_R2=0.5417, val_RMSE=37.4810, time=7.90s, max_mem=152.55MB
  batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=huber, wd=0.0, feat=raw => val_R2=0.5147, val_RMSE=38.5732, time=7.78s, max_mem=153.48MB
  batch=64, epochs=10, lr=0.00050, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=1e-05, feat=raw => val_R2=0.5052, val_RMSE=38.9479, time=7.81s, max_mem=153.48MB

### Selected Config
- batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=log_skew

### Final Test Result (pricing_test.csv)
- R2=0.4584, RMSE=37.2903, TrainTime=45.18s, MaxMem=135.04MB

---
## 2026-02-13 13:06:52

### Sweep Results (train/validation on pricing.csv only)
- Coarse Sweep:
  batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=raw, sku_scale=1.0 => val_R2=0.5297, val_RMSE=37.9700, time=7.83s, max_mem=126.25MB
  batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=log_skew, sku_scale=1.0 => val_R2=0.5431, val_RMSE=37.4241, time=7.92s, max_mem=149.98MB
  batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=raw, sku_scale=0.5 => val_R2=0.5224, val_RMSE=38.2646, time=7.86s, max_mem=151.66MB
  batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=raw, sku_scale=0.2 => val_R2=0.5218, val_RMSE=38.2867, time=7.85s, max_mem=150.95MB
  batch=96, epochs=10, lr=0.00050, log_target=False, clip=10.0, hidden=(160, 96, 48), loss=mse, wd=1e-05, feat=raw, sku_scale=0.5 => val_R2=0.5025, val_RMSE=39.0531, time=8.89s, max_mem=152.04MB
  batch=128, epochs=10, lr=0.00070, log_target=False, clip=10.0, hidden=(160, 96, 48), loss=huber, wd=1e-05, feat=raw, sku_scale=0.5 => val_R2=0.4976, val_RMSE=39.2460, time=8.03s, max_mem=153.21MB
- Focused Sweep:
  batch=48, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=raw, sku_scale=1.0 => val_R2=0.5378, val_RMSE=37.6431, time=7.62s, max_mem=151.63MB
  batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=raw, sku_scale=0.5 => val_R2=0.5203, val_RMSE=38.3485, time=8.01s, max_mem=151.88MB
  batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=raw, sku_scale=0.2 => val_R2=0.5233, val_RMSE=38.2270, time=7.84s, max_mem=154.09MB
  batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=raw, sku_scale=0.0 => val_R2=0.5204, val_RMSE=38.3425, time=7.91s, max_mem=154.09MB
  batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=log_skew, sku_scale=0.5 => val_R2=0.5411, val_RMSE=37.5094, time=8.13s, max_mem=154.09MB
  batch=64, epochs=12, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=raw, sku_scale=0.2 => val_R2=0.5265, val_RMSE=38.0995, time=9.72s, max_mem=152.25MB

### Selected Config
- batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=log_skew, sku_scale=1.0

### Final Test Result (pricing_test.csv)
- R2=0.4584, RMSE=37.2903, TrainTime=44.79s, MaxMem=135.25MB

---
## 2026-02-13 13:11:01

Validation Strategy: sku_shift

### Sweep Results (train/validation on pricing.csv only)
- Coarse Sweep:
  batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=raw, sku_scale=1.0 => val_R2=0.4983, val_RMSE=51.1307, time=7.87s, max_mem=126.07MB
  batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=log_skew, sku_scale=1.0 => val_R2=0.5394, val_RMSE=48.9892, time=8.29s, max_mem=151.15MB
  batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=raw, sku_scale=0.5 => val_R2=0.5022, val_RMSE=50.9324, time=8.19s, max_mem=150.09MB
  batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=raw, sku_scale=0.2 => val_R2=0.5043, val_RMSE=50.8241, time=8.20s, max_mem=151.95MB
  batch=96, epochs=10, lr=0.00050, log_target=False, clip=10.0, hidden=(160, 96, 48), loss=mse, wd=1e-05, feat=raw, sku_scale=0.5 => val_R2=0.4972, val_RMSE=51.1875, time=8.82s, max_mem=151.86MB
  batch=128, epochs=10, lr=0.00070, log_target=False, clip=10.0, hidden=(160, 96, 48), loss=huber, wd=1e-05, feat=raw, sku_scale=0.5 => val_R2=0.4621, val_RMSE=52.9448, time=8.07s, max_mem=153.55MB
- Focused Sweep:
  batch=48, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=raw, sku_scale=1.0 => val_R2=0.5007, val_RMSE=51.0083, time=7.59s, max_mem=151.49MB
  batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=raw, sku_scale=0.5 => val_R2=0.4995, val_RMSE=51.0717, time=7.74s, max_mem=151.37MB
  batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=raw, sku_scale=0.2 => val_R2=0.5018, val_RMSE=50.9502, time=7.64s, max_mem=151.78MB
  batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=raw, sku_scale=0.0 => val_R2=0.5005, val_RMSE=51.0172, time=7.72s, max_mem=153.61MB
  batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=log_skew, sku_scale=0.5 => val_R2=0.5414, val_RMSE=48.8867, time=7.66s, max_mem=151.78MB
  batch=64, epochs=12, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=raw, sku_scale=0.2 => val_R2=0.4800, val_RMSE=52.0561, time=9.55s, max_mem=153.61MB

### Selected Config
- batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=log_skew, sku_scale=0.5

### Final Test Result (pricing_test.csv)
- R2=0.4731, RMSE=36.7834, TrainTime=43.52s, MaxMem=135.57MB

---
## 2026-02-13 13:15:51

Validation Strategy: sku_shift

### Sweep Results (train/validation on pricing.csv only)
- Coarse Sweep:
  batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=log_skew, sku_scale=0.5 => val_R2=0.5397, val_RMSE=48.9754, time=7.73s, max_mem=125.74MB
  batch=64, epochs=10, lr=0.00060, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=log_skew, sku_scale=0.5 => val_R2=0.5390, val_RMSE=49.0153, time=8.64s, max_mem=149.42MB
  batch=64, epochs=10, lr=0.00080, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=log_skew, sku_scale=0.5 => val_R2=0.5439, val_RMSE=48.7497, time=8.16s, max_mem=150.79MB
  batch=48, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=log_skew, sku_scale=0.5 => val_R2=0.5449, val_RMSE=48.6995, time=7.87s, max_mem=150.29MB
  batch=80, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=log_skew, sku_scale=0.5 => val_R2=0.5375, val_RMSE=49.0931, time=7.24s, max_mem=150.35MB
  batch=64, epochs=12, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=log_skew, sku_scale=0.5 => val_R2=0.5338, val_RMSE=49.2868, time=8.92s, max_mem=152.18MB
- Focused Sweep:
  batch=48, epochs=10, lr=0.00060, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=log_skew, sku_scale=0.5 => val_R2=0.5402, val_RMSE=48.9500, time=7.44s, max_mem=150.63MB
  batch=48, epochs=12, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=log_skew, sku_scale=0.5 => val_R2=0.5313, val_RMSE=49.4225, time=8.64s, max_mem=151.56MB
  batch=64, epochs=10, lr=0.00070, log_target=False, clip=7.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=log_skew, sku_scale=0.5 => val_R2=0.5304, val_RMSE=49.4658, time=7.07s, max_mem=150.61MB
  batch=64, epochs=10, lr=0.00070, log_target=False, clip=9.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=log_skew, sku_scale=0.5 => val_R2=0.5442, val_RMSE=48.7357, time=7.18s, max_mem=152.93MB
  batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=log_skew, sku_scale=0.4 => val_R2=0.5421, val_RMSE=48.8471, time=7.17s, max_mem=152.94MB
  batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=log_skew, sku_scale=0.8 => val_R2=0.5411, val_RMSE=48.9023, time=7.12s, max_mem=151.96MB

### Selected Config
- batch=48, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=log_skew, sku_scale=0.5

### Final Test Result (pricing_test.csv)
- R2=0.4669, RMSE=36.9996, TrainTime=40.75s, MaxMem=136.16MB

---
## 2026-02-13 13:24:06

Validation Strategy: sku_shift

### Sweep Results (train/validation on pricing.csv only)
- Coarse Sweep:
  batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=hash_sku, sku_scale=1.0, hash_buckets=16 => val_R2=0.5370, val_RMSE=49.1187, time=7.80s, max_mem=164.56MB
  batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=hash_sku, sku_scale=1.0, hash_buckets=32 => val_R2=0.5375, val_RMSE=49.0928, time=8.02s, max_mem=229.44MB
  batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=hash_sku, sku_scale=1.0, hash_buckets=64 => val_R2=0.5226, val_RMSE=49.8769, time=9.04s, max_mem=312.06MB
  batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=log_skew, sku_scale=0.5, hash_buckets=32 => val_R2=0.5394, val_RMSE=48.9914, time=7.27s, max_mem=152.44MB
  batch=64, epochs=10, lr=0.00060, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=log_skew, sku_scale=0.5, hash_buckets=32 => val_R2=0.5397, val_RMSE=48.9753, time=7.22s, max_mem=152.45MB
  batch=64, epochs=10, lr=0.00080, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=log_skew, sku_scale=0.5, hash_buckets=32 => val_R2=0.5419, val_RMSE=48.8576, time=7.17s, max_mem=151.23MB
- Focused Sweep:
  batch=48, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=hash_sku, sku_scale=1.0, hash_buckets=16 => val_R2=0.5395, val_RMSE=48.9836, time=7.77s, max_mem=190.04MB
  batch=48, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=hash_sku, sku_scale=1.0, hash_buckets=32 => val_R2=0.5376, val_RMSE=49.0862, time=8.18s, max_mem=231.22MB
  batch=48, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=hash_sku, sku_scale=1.0, hash_buckets=64 => val_R2=0.5240, val_RMSE=49.8028, time=9.04s, max_mem=313.27MB
  batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=hash_sku, sku_scale=0.7, hash_buckets=32 => val_R2=0.5437, val_RMSE=48.7599, time=7.95s, max_mem=230.88MB
  batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=hash_sku, sku_scale=1.3, hash_buckets=32 => val_R2=0.5401, val_RMSE=48.9517, time=8.16s, max_mem=230.96MB
  batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=log_skew, sku_scale=0.5, hash_buckets=32 => val_R2=0.5422, val_RMSE=48.8428, time=7.25s, max_mem=152.41MB

### Selected Config
- batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=hash_sku, sku_scale=0.7, hash_buckets=32

### Final Test Result (pricing_test.csv)
- R2=0.4646, RMSE=37.0790, TrainTime=48.13s, MaxMem=139.98MB

---
## 2026-02-13 14:11:17

Validation Strategy: sku_shift

### Sweep Results (train/validation on pricing.csv only)
- Coarse Sweep:
  batch=64, epochs=10, lr=0.00080, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=False => val_R2=0.5403±0.0018, val_RMSE=48.9427±0.0972, time=7.30s, max_mem=141.71MB
  batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=hash_ids, sku_scale=0.7, cat_scale=1.0, hash_buckets=32/16, interact=True => val_R2=0.5092±0.0024, val_RMSE=50.5727±0.1249, time=8.49s, max_mem=266.85MB
  batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=hash_ids, sku_scale=0.7, cat_scale=1.0, hash_buckets=64/16, interact=True => val_R2=0.4971±0.0025, val_RMSE=51.1937±0.1260, time=9.50s, max_mem=344.23MB
  batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=hash_ids, sku_scale=0.7, cat_scale=1.0, hash_buckets=32/32, interact=True => val_R2=0.5094±0.0021, val_RMSE=50.5604±0.1071, time=9.25s, max_mem=306.45MB
  batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=hash_ids, sku_scale=0.7, cat_scale=1.0, hash_buckets=32/16, interact=False => val_R2=0.5486±0.0021, val_RMSE=48.5001±0.1146, time=8.40s, max_mem=260.48MB
  batch=64, epochs=10, lr=0.00080, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=hash_ids, sku_scale=0.7, cat_scale=1.0, hash_buckets=32/16, interact=True => val_R2=0.5129±0.0028, val_RMSE=50.3793±0.1460, time=8.49s, max_mem=267.61MB
- Focused Sweep:
  batch=48, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=hash_ids, sku_scale=0.7, cat_scale=1.0, hash_buckets=32/16, interact=True => val_R2=0.5125±0.0021, val_RMSE=50.4026±0.1075, time=8.72s, max_mem=268.73MB
  batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=hash_ids, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=True => val_R2=0.5090±0.0026, val_RMSE=50.5811±0.1324, time=8.49s, max_mem=269.12MB
  batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=hash_ids, sku_scale=0.9, cat_scale=1.0, hash_buckets=32/16, interact=True => val_R2=0.5083±0.0025, val_RMSE=50.6195±0.1263, time=8.57s, max_mem=267.91MB
  batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=hash_ids, sku_scale=0.7, cat_scale=0.7, hash_buckets=32/16, interact=True => val_R2=0.5080±0.0024, val_RMSE=50.6351±0.1230, time=8.55s, max_mem=268.30MB
  batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=hash_ids, sku_scale=0.7, cat_scale=1.3, hash_buckets=32/16, interact=True => val_R2=0.5083±0.0034, val_RMSE=50.6177±0.1731, time=8.54s, max_mem=269.25MB
  batch=64, epochs=10, lr=0.00080, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=hash_ids, sku_scale=0.7, cat_scale=1.0, hash_buckets=32/16, interact=True => val_R2=0.5129±0.0028, val_RMSE=50.3793±0.1460, time=8.49s, max_mem=268.41MB

### Selected Config
- batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=hash_ids, sku_scale=0.7, cat_scale=1.0, hash_buckets=32/16, interact=False

### Final Test Result (pricing_test.csv)
- R2=0.4660, RMSE=37.0278, TrainTime=49.88s, MaxMem=143.70MB

---
## 2026-02-13 14:44:29

Validation Strategy: sku_shift

### Sweep Results (train/validation on pricing.csv only)
- Coarse Sweep:
  batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=hash_ids, sku_scale=0.7, cat_scale=1.0, hash_buckets=32/16, interact=False => val_R2=0.5677±0.0013, val_RMSE=28.7548±0.0432, time=35.47s, max_mem=730.61MB
  batch=64, epochs=10, lr=0.00080, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=hash_ids, sku_scale=0.7, cat_scale=1.0, hash_buckets=32/16, interact=False => val_R2=0.5653±0.0015, val_RMSE=28.8345±0.0486, time=34.84s, max_mem=755.71MB
  batch=64, epochs=10, lr=0.00070, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=False => val_R2=0.5725±0.0010, val_RMSE=28.5962±0.0340, time=29.96s, max_mem=297.50MB
  batch=64, epochs=10, lr=0.00080, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=False => val_R2=0.5740±0.0015, val_RMSE=28.5442±0.0503, time=29.90s, max_mem=297.32MB
- Focused Sweep:
  batch=48, epochs=10, lr=0.00080, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=hash_ids, sku_scale=0.7, cat_scale=1.0, hash_buckets=32/16, interact=False => val_R2=0.5611±0.0012, val_RMSE=28.9743±0.0388, time=35.79s, max_mem=755.55MB
  batch=64, epochs=10, lr=0.00080, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=hash_ids, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=False => val_R2=0.5676±0.0025, val_RMSE=28.7601±0.0832, time=34.78s, max_mem=754.35MB
  batch=64, epochs=10, lr=0.00080, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=hash_ids, sku_scale=0.9, cat_scale=1.0, hash_buckets=32/16, interact=False => val_R2=0.5648±0.0011, val_RMSE=28.8521±0.0378, time=34.71s, max_mem=754.32MB
  batch=64, epochs=10, lr=0.00080, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=hash_ids, sku_scale=0.7, cat_scale=0.8, hash_buckets=32/16, interact=False => val_R2=0.5675±0.0022, val_RMSE=28.7618±0.0718, time=34.71s, max_mem=754.29MB
  batch=64, epochs=10, lr=0.00080, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=hash_ids, sku_scale=0.7, cat_scale=1.2, hash_buckets=32/16, interact=False => val_R2=0.5654±0.0024, val_RMSE=28.8331±0.0797, time=34.74s, max_mem=754.51MB
  batch=64, epochs=10, lr=0.00080, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=False => val_R2=0.5740±0.0015, val_RMSE=28.5442±0.0503, time=30.00s, max_mem=297.43MB

### Selected Config
- batch=64, epochs=10, lr=0.00080, log_target=False, clip=8.0, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=False

### Final Test Result (pricing_test.csv)
- R2=0.4834, RMSE=36.4215, TrainTime=40.78s, MaxMem=207.82MB

---
## 2026-02-13 19:45:11

Validation Strategy: sku_shift

### Sweep Results (train/validation on pricing.csv only)
- Coarse Sweep:
  batch=48, epochs=22, lr=0.00070, log_target=False, clip=9.0, decay=0.9930, min_lr=0.000010, hidden=(192, 96, 48), loss=mse, wd=0.0, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=True => val_R2=0.5827+/-0.0011, val_RMSE=28.2538+/-0.0370, time=131.76s, max_mem=309.08MB
  batch=48, epochs=24, lr=0.00060, log_target=False, clip=10.0, decay=0.9940, min_lr=0.000010, hidden=(256, 128, 64), loss=mse, wd=0.0, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=True => val_R2=0.5743+/-0.0033, val_RMSE=28.5344+/-0.1095, time=186.55s, max_mem=353.06MB
  batch=48, epochs=24, lr=0.00070, log_target=False, clip=9.0, decay=0.9940, min_lr=0.000010, hidden=(192, 96, 48), loss=mse, wd=0.0, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=False => val_R2=0.5794+/-0.0026, val_RMSE=28.3630+/-0.0893, time=125.16s, max_mem=301.82MB
- Focused Sweep:
  batch=32, epochs=28, lr=0.00060, log_target=False, clip=9.0, decay=0.9950, min_lr=0.000010, hidden=(192, 96, 48), loss=mse, wd=0.0, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=True => val_R2=0.5807+/-0.0009, val_RMSE=28.3220+/-0.0296, time=194.36s, max_mem=370.03MB
  batch=32, epochs=30, lr=0.00050, log_target=False, clip=10.0, decay=0.9960, min_lr=0.000010, hidden=(256, 128, 64), loss=mse, wd=0.0, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=True => val_R2=0.5623+/-0.0038, val_RMSE=28.9350+/-0.1243, time=309.90s, max_mem=373.67MB
  batch=48, epochs=24, lr=0.00070, log_target=False, clip=8.0, decay=0.9940, min_lr=0.000010, hidden=(192, 96, 48), loss=mse, wd=0.0, feat=hash_ids, sku_scale=0.7, cat_scale=1.0, hash_buckets=32/16, interact=False => val_R2=0.5517+/-0.0010, val_RMSE=29.2846+/-0.0325, time=144.62s, max_mem=739.00MB

### Selected Config
- batch=48, epochs=22, lr=0.00070, log_target=False, clip=9.0, decay=0.9930, min_lr=0.000010, hidden=(192, 96, 48), loss=mse, wd=0.0, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.4569, RMSE=37.3425, TrainTime=155.28s, MaxMem=192.56MB

---
## 2026-02-13 22:09:58

Validation Strategy: sku_shift

### Sweep Results (train/validation on pricing.csv only)
- Coarse Sweep:
  batch=48, epochs=22, lr=0.00070, log_target=False, clip=9.0, decay=0.9930, min_lr=0.000010, hidden=(192, 96, 48), loss=mse, wd=0.0, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=True => val_R2=0.5827+/-0.0011, val_RMSE=28.2538+/-0.0370, time=124.18s, max_mem=309.38MB
  batch=48, epochs=24, lr=0.00060, log_target=False, clip=10.0, decay=0.9940, min_lr=0.000010, hidden=(256, 128, 64), loss=mse, wd=0.0, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=True => val_R2=0.5743+/-0.0033, val_RMSE=28.5344+/-0.1095, time=291.59s, max_mem=353.47MB
  batch=48, epochs=24, lr=0.00070, log_target=False, clip=9.0, decay=0.9940, min_lr=0.000010, hidden=(192, 96, 48), loss=mse, wd=0.0, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=False => val_R2=0.5794+/-0.0026, val_RMSE=28.3630+/-0.0893, time=215.34s, max_mem=280.54MB
- Focused Sweep:
  batch=32, epochs=28, lr=0.00060, log_target=False, clip=9.0, decay=0.9950, min_lr=0.000010, hidden=(192, 96, 48), loss=mse, wd=0.0, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=True => val_R2=0.5807+/-0.0009, val_RMSE=28.3220+/-0.0296, time=588.70s, max_mem=313.47MB
  batch=32, epochs=30, lr=0.00050, log_target=False, clip=10.0, decay=0.9960, min_lr=0.000010, hidden=(256, 128, 64), loss=mse, wd=0.0, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=True => val_R2=0.5623+/-0.0038, val_RMSE=28.9350+/-0.1243, time=963.29s, max_mem=279.01MB
  batch=48, epochs=24, lr=0.00070, log_target=False, clip=8.0, decay=0.9940, min_lr=0.000010, hidden=(192, 96, 48), loss=mse, wd=0.0, feat=hash_ids, sku_scale=0.7, cat_scale=1.0, hash_buckets=32/16, interact=False => val_R2=0.5517+/-0.0010, val_RMSE=29.2846+/-0.0325, time=395.73s, max_mem=678.17MB

### Selected Config
- batch=48, epochs=22, lr=0.00070, log_target=False, clip=9.0, decay=0.9930, min_lr=0.000010, hidden=(192, 96, 48), loss=mse, wd=0.0, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.4720, RMSE=36.8193, TrainTime=647.09s, MaxMem=182.62MB

---
## 2026-02-17 10:56:23

Validation Strategy: sku_shift

### Sweep Results (train/validation on pricing.csv only)
- Coarse Sweep:
  batch=48, epochs=22, lr=0.00070, log_target=False, clip=9.0, decay=0.9930, min_lr=0.000010, hidden=(192, 96, 48), loss=mse, wd=0.0, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=True => val_R2=0.5827+/-0.0011, val_RMSE=28.2538+/-0.0370, time=129.09s, max_mem=308.90MB
  batch=48, epochs=24, lr=0.00060, log_target=False, clip=10.0, decay=0.9940, min_lr=0.000010, hidden=(256, 128, 64), loss=mse, wd=0.0, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=True => val_R2=0.5743+/-0.0033, val_RMSE=28.5344+/-0.1095, time=191.69s, max_mem=352.08MB
  batch=48, epochs=24, lr=0.00070, log_target=False, clip=9.0, decay=0.9940, min_lr=0.000010, hidden=(192, 96, 48), loss=mse, wd=0.0, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=False => val_R2=0.5794+/-0.0026, val_RMSE=28.3630+/-0.0893, time=124.94s, max_mem=232.63MB
- Focused Sweep:
  batch=32, epochs=28, lr=0.00060, log_target=False, clip=9.0, decay=0.9950, min_lr=0.000010, hidden=(192, 96, 48), loss=mse, wd=0.0, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=True => val_R2=0.5807+/-0.0009, val_RMSE=28.3220+/-0.0296, time=195.13s, max_mem=323.99MB
  batch=32, epochs=30, lr=0.00050, log_target=False, clip=10.0, decay=0.9960, min_lr=0.000010, hidden=(256, 128, 64), loss=mse, wd=0.0, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=True => val_R2=0.5623+/-0.0038, val_RMSE=28.9350+/-0.1243, time=295.59s, max_mem=324.41MB
  batch=48, epochs=24, lr=0.00070, log_target=False, clip=8.0, decay=0.9940, min_lr=0.000010, hidden=(192, 96, 48), loss=mse, wd=0.0, feat=hash_ids, sku_scale=0.7, cat_scale=1.0, hash_buckets=32/16, interact=False => val_R2=0.5517+/-0.0010, val_RMSE=29.2846+/-0.0325, time=143.41s, max_mem=715.66MB

### Selected Config
- batch=48, epochs=22, lr=0.00070, log_target=False, clip=9.0, decay=0.9930, min_lr=0.000010, hidden=(192, 96, 48), loss=mse, wd=0.0, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.4720, RMSE=36.8193, TrainTime=162.24s, MaxMem=196.94MB

---
## 2026-02-17 12:11:36

Validation Strategy: sku_shift

### Sweep Results (train/validation on pricing.csv only)
- Coarse Sweep:
  batch=48, epochs=22, lr=0.00070, log_target=False, clip=9.0, decay=0.9930, min_lr=0.000010, hidden=(192, 96, 48), loss=mse, wd=0.0, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=True => val_R2=0.5827+/-0.0011, val_RMSE=28.2538+/-0.0370, time=118.55s, max_mem=416.21MB
  batch=48, epochs=22, lr=0.00070, log_target=False, clip=9.0, decay=0.9930, min_lr=0.000010, hidden=(192, 96, 48), loss=mse, wd=1e-06, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=True => val_R2=0.5819+/-0.0015, val_RMSE=28.2795+/-0.0517, time=121.84s, max_mem=415.06MB
  batch=48, epochs=22, lr=0.00070, log_target=False, clip=9.0, decay=0.9930, min_lr=0.000010, hidden=(192, 96, 48), loss=mse, wd=3e-06, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=True => val_R2=0.5821+/-0.0012, val_RMSE=28.2727+/-0.0408, time=122.56s, max_mem=415.28MB
  batch=48, epochs=22, lr=0.00070, log_target=False, clip=9.0, decay=0.9930, min_lr=0.000010, hidden=(192, 96, 48), loss=mse, wd=1e-05, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=True => val_R2=0.5815+/-0.0005, val_RMSE=28.2944+/-0.0166, time=121.63s, max_mem=415.41MB
  batch=48, epochs=22, lr=0.00070, log_target=False, clip=9.0, decay=0.9930, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=True => val_R2=0.5823+/-0.0013, val_RMSE=28.2670+/-0.0450, time=101.90s, max_mem=415.27MB
  batch=48, epochs=22, lr=0.00070, log_target=False, clip=9.0, decay=0.9930, min_lr=0.000010, hidden=(128, 64, 32), loss=mse, wd=1e-06, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=True => val_R2=0.5834+/-0.0017, val_RMSE=28.2291+/-0.0581, time=70.29s, max_mem=417.26MB
- Focused Sweep:
  batch=48, epochs=18, lr=0.00070, log_target=False, clip=9.0, decay=0.9930, min_lr=0.000010, hidden=(192, 96, 48), loss=mse, wd=1e-06, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=True => val_R2=0.5430+/-0.0021, val_RMSE=29.5677+/-0.0693, time=100.55s, max_mem=414.64MB
  batch=48, epochs=22, lr=0.00070, log_target=False, clip=9.0, decay=0.9930, min_lr=0.000010, hidden=(192, 96, 48), loss=mse, wd=1e-06, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=True => val_R2=0.5819+/-0.0015, val_RMSE=28.2795+/-0.0517, time=121.55s, max_mem=414.79MB
  batch=48, epochs=26, lr=0.00070, log_target=False, clip=9.0, decay=0.9930, min_lr=0.000010, hidden=(192, 96, 48), loss=mse, wd=1e-06, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=True => val_R2=0.5829+/-0.0010, val_RMSE=28.2459+/-0.0333, time=143.59s, max_mem=414.80MB
  batch=48, epochs=18, lr=0.00070, log_target=False, clip=9.0, decay=0.9930, min_lr=0.000010, hidden=(192, 96, 48), loss=mse, wd=3e-06, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=True => val_R2=0.5447+/-0.0013, val_RMSE=29.5114+/-0.0424, time=100.43s, max_mem=414.80MB
  batch=48, epochs=22, lr=0.00070, log_target=False, clip=9.0, decay=0.9930, min_lr=0.000010, hidden=(192, 96, 48), loss=mse, wd=3e-06, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=True => val_R2=0.5821+/-0.0012, val_RMSE=28.2727+/-0.0408, time=125.79s, max_mem=414.84MB
  batch=48, epochs=26, lr=0.00070, log_target=False, clip=9.0, decay=0.9930, min_lr=0.000010, hidden=(192, 96, 48), loss=mse, wd=3e-06, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=True => val_R2=0.5816+/-0.0008, val_RMSE=28.2890+/-0.0283, time=143.73s, max_mem=414.84MB

### Selected Config
- batch=48, epochs=22, lr=0.00070, log_target=False, clip=9.0, decay=0.9930, min_lr=0.000010, hidden=(128, 64, 32), loss=mse, wd=1e-06, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.4627, RMSE=37.1433, TrainTime=95.09s, MaxMem=290.78MB

---
## 2026-02-17 17:12:41

Validation Strategy: ['sku_shift', 'random']
Selection Rule: mean(val_R2) - 0.50*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)
- Coarse Sweep:
  batch=48, epochs=22, lr=0.00070, log_target=False, clip=9.0, decay=0.9930, min_lr=0.000010, hidden=(192, 96, 48), loss=mse, wd=0.0, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=True => score=0.5810, val_R2=0.5817+/-0.0013, val_RMSE=30.2146+/-1.9611, time=121.80s, max_mem=483.29MB
  splits: sku_shift:R2=0.5827+/-0.0011; random:R2=0.5806+/-0.0006
  batch=48, epochs=22, lr=0.00070, log_target=False, clip=9.0, decay=0.9930, min_lr=0.000010, hidden=(192, 96, 48), loss=mse, wd=1e-06, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=True => score=0.5802, val_R2=0.5809+/-0.0015, val_RMSE=30.2409+/-1.9618, time=126.01s, max_mem=452.71MB
  splits: sku_shift:R2=0.5819+/-0.0015; random:R2=0.5799+/-0.0003
  batch=48, epochs=22, lr=0.00070, log_target=False, clip=9.0, decay=0.9930, min_lr=0.000010, hidden=(192, 96, 48), loss=mse, wd=3e-06, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=True => score=0.5807, val_R2=0.5813+/-0.0012, val_RMSE=30.2258+/-1.9534, time=124.73s, max_mem=365.08MB
  splits: sku_shift:R2=0.5821+/-0.0012; random:R2=0.5805+/-0.0004
  batch=48, epochs=22, lr=0.00070, log_target=False, clip=9.0, decay=0.9930, min_lr=0.000010, hidden=(192, 96, 48), loss=mse, wd=1e-05, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=True => score=0.5804, val_R2=0.5809+/-0.0009, val_RMSE=30.2423+/-1.9480, time=127.18s, max_mem=320.25MB
  splits: sku_shift:R2=0.5815+/-0.0005; random:R2=0.5803+/-0.0007
  batch=48, epochs=22, lr=0.00070, log_target=False, clip=9.0, decay=0.9930, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=True => score=0.5798, val_R2=0.5807+/-0.0019, val_RMSE=30.2504+/-1.9836, time=115.28s, max_mem=283.56MB
  splits: sku_shift:R2=0.5823+/-0.0013; random:R2=0.5791+/-0.0002
  batch=48, epochs=22, lr=0.00070, log_target=False, clip=9.0, decay=0.9930, min_lr=0.000010, hidden=(128, 64, 32), loss=mse, wd=1e-06, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=True => score=0.5805, val_R2=0.5817+/-0.0024, val_RMSE=30.2133+/-1.9851, time=71.95s, max_mem=298.89MB
  splits: sku_shift:R2=0.5834+/-0.0017; random:R2=0.5801+/-0.0017
- Focused Sweep:
  batch=48, epochs=18, lr=0.00070, log_target=False, clip=9.0, decay=0.9930, min_lr=0.000010, hidden=(192, 96, 48), loss=mse, wd=1e-06, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=True => score=0.5504, val_R2=0.5582+/-0.0156, val_RMSE=31.0086+/-1.4462, time=100.37s, max_mem=325.03MB
  splits: sku_shift:R2=0.5430+/-0.0021; random:R2=0.5735+/-0.0042
  batch=48, epochs=22, lr=0.00070, log_target=False, clip=9.0, decay=0.9930, min_lr=0.000010, hidden=(192, 96, 48), loss=mse, wd=1e-06, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=True => score=0.5802, val_R2=0.5809+/-0.0015, val_RMSE=30.2409+/-1.9618, time=124.82s, max_mem=325.42MB
  splits: sku_shift:R2=0.5819+/-0.0015; random:R2=0.5799+/-0.0003
  batch=48, epochs=26, lr=0.00070, log_target=False, clip=9.0, decay=0.9930, min_lr=0.000010, hidden=(192, 96, 48), loss=mse, wd=1e-06, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=True => score=0.5828, val_R2=0.5838+/-0.0021, val_RMSE=30.1324+/-1.8879, time=144.15s, max_mem=327.81MB
  splits: sku_shift:R2=0.5829+/-0.0010; random:R2=0.5847+/-0.0025
  batch=48, epochs=18, lr=0.00070, log_target=False, clip=9.0, decay=0.9930, min_lr=0.000010, hidden=(192, 96, 48), loss=mse, wd=3e-06, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=True => score=0.5514, val_R2=0.5583+/-0.0138, val_RMSE=31.0113+/-1.5030, time=99.76s, max_mem=326.79MB
  splits: sku_shift:R2=0.5447+/-0.0013; random:R2=0.5718+/-0.0035
  batch=48, epochs=22, lr=0.00070, log_target=False, clip=9.0, decay=0.9930, min_lr=0.000010, hidden=(192, 96, 48), loss=mse, wd=3e-06, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=True => score=0.5807, val_R2=0.5813+/-0.0012, val_RMSE=30.2258+/-1.9534, time=133.89s, max_mem=327.27MB
  splits: sku_shift:R2=0.5821+/-0.0012; random:R2=0.5805+/-0.0004
  batch=48, epochs=26, lr=0.00070, log_target=False, clip=9.0, decay=0.9930, min_lr=0.000010, hidden=(192, 96, 48), loss=mse, wd=3e-06, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=True => score=0.5822, val_R2=0.5832+/-0.0020, val_RMSE=30.1537+/-1.8653, time=156.12s, max_mem=296.56MB
  splits: sku_shift:R2=0.5816+/-0.0008; random:R2=0.5847+/-0.0017

### Selected Config
- batch=48, epochs=26, lr=0.00070, log_target=False, clip=9.0, decay=0.9930, min_lr=0.000010, hidden=(192, 96, 48), loss=mse, wd=1e-06, feat=log_skew, sku_scale=0.5, cat_scale=1.0, score=0.5828, hash_buckets=32/16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.4201, RMSE=38.5874, TrainTime=207.44s, MaxMem=165.58MB

---
## 2026-02-17 19:30:12

Validation Strategy: ['sku_shift', 'random']
Selection Rule: mean(val_R2) - 0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)
- Coarse Sweep:
  batch=48, epochs=22, lr=0.00070, log_target=False, clip=9.0, decay=0.9930, min_lr=0.000010, hidden=(192, 96, 48), loss=mse, wd=1e-06, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=24/12, interact=True => score=0.5541, val_R2=0.5697+/-0.0222, val_RMSE=36.7808+/-2.1419, time=44.38s, max_mem=190.99MB
  splits: sku_shift:R2=0.5475+/-0.0006; random:R2=0.5919+/-0.0007
  batch=48, epochs=20, lr=0.00070, log_target=False, clip=9.0, decay=0.9930, min_lr=0.000010, hidden=(128, 64, 32), loss=mse, wd=1e-06, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=24/12, interact=True => score=0.5040, val_R2=0.5360+/-0.0456, val_RMSE=38.1906+/-3.1155, time=23.02s, max_mem=203.29MB
  splits: sku_shift:R2=0.4904+/-0.0017; random:R2=0.5815+/-0.0017
  batch=48, epochs=20, lr=0.00070, log_target=False, clip=9.0, decay=0.9930, min_lr=0.000010, hidden=(128, 64, 32), loss=mse, wd=1e-06, feat=log_skew_fe, sku_scale=0.35, cat_scale=1.0, hash_buckets=24/12, interact=True => score=0.5175, val_R2=0.5457+/-0.0403, val_RMSE=37.7906+/-2.9003, time=26.33s, max_mem=333.56MB
  splits: sku_shift:R2=0.5055+/-0.0027; random:R2=0.5859+/-0.0006

### Selected Config
- batch=48, epochs=22, lr=0.00070, log_target=False, clip=9.0, decay=0.9930, min_lr=0.000010, hidden=(192, 96, 48), loss=mse, wd=1e-06, feat=log_skew, sku_scale=0.5, cat_scale=1.0, score=0.5541, hash_buckets=24/12, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.4343, RMSE=38.1112, TrainTime=158.58s, MaxMem=183.66MB

---
## 2026-02-17 19:32:39

Validation Strategy: ['sku_shift']
Selection Rule: mean(val_R2) - 0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)

### Selected Config
- batch=64, epochs=10, lr=0.00080, log_target=False, clip=8.0, decay=1.0000, min_lr=0.000010, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=log_skew, sku_scale=0.5, cat_scale=1.0, score=nan, hash_buckets=32/16, interact=False

### Final Test Result (pricing_test.csv)
- R2=0.4517, RMSE=37.5204, TrainTime=40.50s, MaxMem=135.46MB

---
## 2026-02-17 19:34:04

Validation Strategy: ['sku_shift']
Selection Rule: mean(val_R2) - 0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)

### Selected Config
- batch=64, epochs=10, lr=0.00080, log_target=False, clip=8.0, decay=1.0000, min_lr=0.000010, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=log_skew_fe, sku_scale=0.35, cat_scale=1.0, score=nan, hash_buckets=32/16, interact=False

### Final Test Result (pricing_test.csv)
- R2=0.4666, RMSE=37.0070, TrainTime=49.76s, MaxMem=158.03MB

---
## 2026-02-17 19:43:49

Validation Strategy: ['sku_shift']
Selection Rule: mean(val_R2) - 0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)

### Selected Config
- batch=64, epochs=10, lr=0.00080, log_target=False, clip=8.0, decay=1.0000, min_lr=0.000010, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=log_skew_fe, sku_scale=0.35, cat_scale=1.0, score=nan, hash_buckets=32/16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.4682, RMSE=36.9542, TrainTime=55.85s, MaxMem=159.45MB

---
## 2026-02-17 19:55:21

Validation Strategy: ['sku_shift']
Selection Rule: mean(val_R2) - 0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)

### Selected Config
- batch=64, epochs=10, lr=0.00080, log_target=False, clip=8.0, decay=1.0000, min_lr=0.000010, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=log_skew_fe, sku_scale=0.35, cat_scale=1.0, score=nan, hash_buckets=32/16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.4682, RMSE=36.9542, TrainTime=50.65s, MaxMem=159.85MB

---
## 2026-02-17 20:14:32

Validation Strategy: ['sku_shift']
Selection Rule: mean(val_R2) - 0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)

### Selected Config
- batch=64, epochs=10, lr=0.00080, log_target=False, clip=8.0, decay=1.0000, min_lr=0.000010, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=log_skew_fe, sku_scale=0.35, cat_scale=1.0, score=nan, hash_buckets=32/16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.4682, RMSE=36.9542, TrainTime=52.82s, MaxMem=158.81MB

---
## 2026-02-17 20:16:35

Validation Strategy: ['sku_shift']
Selection Rule: mean(val_R2) - 0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)

### Selected Config
- batch=64, epochs=10, lr=0.00080, log_target=False, clip=8.0, decay=1.0000, min_lr=0.000010, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=log_skew, sku_scale=0.5, cat_scale=1.0, score=nan, hash_buckets=32/16, interact=False

### Final Test Result (pricing_test.csv)
- R2=0.4517, RMSE=37.5204, TrainTime=41.93s, MaxMem=134.48MB

---
## 2026-02-17 20:18:09

Validation Strategy: ['sku_shift']
Selection Rule: mean(val_R2) - 0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)

### Selected Config
- batch=64, epochs=10, lr=0.00080, log_target=False, clip=8.0, decay=1.0000, min_lr=0.000010, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=log_skew_fe, sku_scale=0.35, cat_scale=1.0, score=nan, hash_buckets=48/24, interact=True

### Final Test Result (pricing_test.csv)
- R2=-3.5749, RMSE=108.3842, TrainTime=57.94s, MaxMem=170.26MB

---
## 2026-02-17 20:19:33

Validation Strategy: ['sku_shift']
Selection Rule: mean(val_R2) - 0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)

### Selected Config
- batch=64, epochs=10, lr=0.00080, log_target=False, clip=8.0, decay=1.0000, min_lr=0.000010, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=log_skew_fe, sku_scale=0.25, cat_scale=0.8, score=nan, hash_buckets=32/16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.4756, RMSE=36.6965, TrainTime=52.50s, MaxMem=159.69MB

---
## 2026-02-17 20:23:23

Validation Strategy: ['sku_shift']
Selection Rule: mean(val_R2) - 0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)

### Selected Config
- batch=64, epochs=10, lr=0.00080, log_target=False, clip=8.0, decay=1.0000, min_lr=0.000010, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=log_skew_fe, sku_scale=0.2, cat_scale=0.7, score=nan, hash_buckets=32/16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.4796, RMSE=36.5539, TrainTime=52.38s, MaxMem=158.27MB

---
## 2026-02-17 20:25:05

Validation Strategy: ['sku_shift']
Selection Rule: mean(val_R2) - 0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)

### Selected Config
- batch=64, epochs=10, lr=0.00080, log_target=False, clip=8.0, decay=1.0000, min_lr=0.000010, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=log_skew_fe, sku_scale=0.15, cat_scale=0.6, score=nan, hash_buckets=32/16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.4797, RMSE=36.5501, TrainTime=52.32s, MaxMem=159.57MB

---
## 2026-02-17 20:26:23

Validation Strategy: ['sku_shift']
Selection Rule: mean(val_R2) - 0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)

### Selected Config
- batch=64, epochs=10, lr=0.00080, log_target=False, clip=8.0, decay=1.0000, min_lr=0.000010, hidden=(128, 64, 32), loss=mse, wd=1e-06, feat=log_skew_fe, sku_scale=0.15, cat_scale=0.6, score=nan, hash_buckets=32/16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.4823, RMSE=36.4598, TrainTime=53.22s, MaxMem=159.00MB

---
## 2026-02-17 20:27:48

Validation Strategy: ['sku_shift']
Selection Rule: mean(val_R2) - 0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)

### Selected Config
- batch=64, epochs=10, lr=0.00080, log_target=False, clip=8.0, decay=1.0000, min_lr=0.000010, hidden=(128, 64, 32), loss=mse, wd=1e-06, feat=log_skew_fe, sku_scale=0.2, cat_scale=0.7, score=nan, hash_buckets=32/16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.4824, RMSE=36.4561, TrainTime=53.29s, MaxMem=157.22MB

---
## 2026-02-17 20:28:59

Validation Strategy: ['sku_shift']
Selection Rule: mean(val_R2) - 0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)

### Selected Config
- batch=64, epochs=10, lr=0.00080, log_target=False, clip=9.0, decay=1.0000, min_lr=0.000010, hidden=(128, 64, 32), loss=mse, wd=1e-06, feat=log_skew_fe, sku_scale=0.2, cat_scale=0.7, score=nan, hash_buckets=32/16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.4817, RMSE=36.4796, TrainTime=53.40s, MaxMem=159.02MB

---
## 2026-02-17 20:30:10

Validation Strategy: ['sku_shift']
Selection Rule: mean(val_R2) - 0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)

### Selected Config
- batch=64, epochs=10, lr=0.00080, log_target=False, clip=7.0, decay=1.0000, min_lr=0.000010, hidden=(128, 64, 32), loss=mse, wd=1e-06, feat=log_skew_fe, sku_scale=0.2, cat_scale=0.7, score=nan, hash_buckets=32/16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.4824, RMSE=36.4567, TrainTime=53.62s, MaxMem=158.93MB

---
## 2026-02-17 21:06:17

Validation Strategy: ['sku_shift', 'random']
Selection Rule: mean(val_R2) - 0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)
- Coarse Sweep:
  batch=64, epochs=10, lr=0.00080, log_target=False, clip=8.0, decay=1.0000, min_lr=0.000010, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=False => score=0.5175, val_R2=0.5259+/-0.0121, val_RMSE=38.6004+/-1.7366, time=12.13s, max_mem=169.52MB
  splits: sku_shift:R2=0.5141+/-0.0032; random:R2=0.5377+/-0.0009
  batch=64, epochs=10, lr=0.00080, log_target=False, clip=8.0, decay=1.0000, min_lr=0.000010, hidden=(128, 64, 32), loss=mse, wd=1e-06, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=False => score=0.5165, val_R2=0.5253+/-0.0125, val_RMSE=38.6264+/-1.7558, time=12.64s, max_mem=178.12MB
  splits: sku_shift:R2=0.5130+/-0.0036; random:R2=0.5375+/-0.0008
  batch=64, epochs=10, lr=0.00080, log_target=False, clip=8.0, decay=1.0000, min_lr=0.000010, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=log_skew_fe, sku_scale=0.2, cat_scale=0.7, hash_buckets=32/16, interact=True => score=0.5390, val_R2=0.5448+/-0.0084, val_RMSE=37.8185+/-1.5446, time=14.74s, max_mem=371.27MB
  splits: sku_shift:R2=0.5374+/-0.0047; random:R2=0.5522+/-0.0027
  batch=64, epochs=10, lr=0.00080, log_target=False, clip=8.0, decay=1.0000, min_lr=0.000010, hidden=(128, 64, 32), loss=mse, wd=1e-06, feat=log_skew_fe, sku_scale=0.2, cat_scale=0.7, hash_buckets=32/16, interact=True => score=0.5387, val_R2=0.5447+/-0.0086, val_RMSE=37.8245+/-1.5538, time=15.25s, max_mem=371.47MB
  splits: sku_shift:R2=0.5370+/-0.0048; random:R2=0.5523+/-0.0027

### Selected Config
- batch=64, epochs=10, lr=0.00080, log_target=False, clip=8.0, decay=1.0000, min_lr=0.000010, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=log_skew_fe, sku_scale=0.2, cat_scale=0.7, score=0.5390, hash_buckets=32/16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.4820, RMSE=36.4696, TrainTime=57.71s, MaxMem=193.06MB

---
## 2026-02-17 21:13:19

Validation Strategy: ['sku_shift', 'random']
Selection Rule: mean(val_R2) - 0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)
- Coarse Sweep:
  batch=64, epochs=10, lr=0.00080, log_target=False, clip=8.0, decay=1.0000, min_lr=0.000010, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=log_skew_fe, sku_scale=0.2, cat_scale=0.7, hash_buckets=32/16, interact=True => score=0.5390, val_R2=0.5448+/-0.0084, val_RMSE=37.8185+/-1.5446, time=15.17s, max_mem=362.91MB
  splits: sku_shift:R2=0.5374+/-0.0047; random:R2=0.5522+/-0.0027
  batch=64, epochs=12, lr=0.00075, log_target=False, clip=8.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_skew_fe, sku_scale=0.2, cat_scale=0.7, hash_buckets=32/16, interact=True => score=0.5639, val_R2=0.5703+/-0.0091, val_RMSE=36.7474+/-1.5659, time=23.61s, max_mem=372.76MB
  splits: sku_shift:R2=0.5617+/-0.0044; random:R2=0.5789+/-0.0001
  batch=64, epochs=12, lr=0.00070, log_target=False, clip=8.0, decay=0.9960, min_lr=0.000010, hidden=(192, 96, 48), loss=mse, wd=1e-06, feat=log_skew_fe, sku_scale=0.2, cat_scale=0.7, hash_buckets=32/16, interact=True => score=0.5614, val_R2=0.5682+/-0.0097, val_RMSE=36.8341+/-1.5878, time=27.41s, max_mem=373.72MB
  splits: sku_shift:R2=0.5592+/-0.0051; random:R2=0.5773+/-0.0008

### Selected Config
- batch=64, epochs=12, lr=0.00075, log_target=False, clip=8.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_skew_fe, sku_scale=0.2, cat_scale=0.7, score=0.5639, hash_buckets=32/16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.4850, RMSE=36.3665, TrainTime=90.96s, MaxMem=200.43MB

---
## 2026-02-17 21:21:59

Validation Strategy: ['sku_shift', 'random']
Selection Rule: mean(val_R2) - 0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)
- Coarse Sweep:
  batch=64, epochs=12, lr=0.00075, log_target=False, clip=8.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_skew_fe, sku_scale=0.2, cat_scale=0.7, hash_buckets=32/16, interact=True => score=0.5639, val_R2=0.5703+/-0.0091, val_RMSE=36.7474+/-1.5659, time=23.01s, max_mem=354.35MB
  splits: sku_shift:R2=0.5617+/-0.0044; random:R2=0.5789+/-0.0001
  batch=64, epochs=12, lr=0.00075, log_target=False, clip=7.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_skew_fe, sku_scale=0.2, cat_scale=0.7, hash_buckets=32/16, interact=True => score=0.5619, val_R2=0.5677+/-0.0084, val_RMSE=36.8551+/-1.5333, time=22.00s, max_mem=360.21MB
  splits: sku_shift:R2=0.5600+/-0.0044; random:R2=0.5755+/-0.0005
  batch=64, epochs=12, lr=0.00075, log_target=False, clip=8.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_skew_fe, sku_scale=0.18, cat_scale=0.65, hash_buckets=32/16, interact=True => score=0.5650, val_R2=0.5711+/-0.0086, val_RMSE=36.7118+/-1.5438, time=22.10s, max_mem=360.01MB
  splits: sku_shift:R2=0.5630+/-0.0043; random:R2=0.5792+/-0.0002
  batch=64, epochs=12, lr=0.00075, log_target=False, clip=7.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_skew_fe, sku_scale=0.18, cat_scale=0.65, hash_buckets=32/16, interact=True => score=0.5622, val_R2=0.5681+/-0.0085, val_RMSE=36.8388+/-1.5295, time=22.16s, max_mem=360.44MB
  splits: sku_shift:R2=0.5605+/-0.0051; random:R2=0.5757+/-0.0007

### Selected Config
- batch=64, epochs=12, lr=0.00075, log_target=False, clip=8.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_skew_fe, sku_scale=0.18, cat_scale=0.65, score=0.5650, hash_buckets=32/16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.4837, RMSE=36.4108, TrainTime=84.65s, MaxMem=180.92MB

---
## 2026-02-17 21:30:46

Validation Strategy: ['sku_shift', 'random']
Selection Rule: mean(val_R2) - 0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)
- Coarse Sweep:
  batch=64, epochs=12, lr=0.00075, log_target=False, clip=8.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_skew_fe, sku_scale=0.2, cat_scale=0.7, hash_buckets=32/16, interact=True => score=0.5639, val_R2=0.5703+/-0.0091, val_RMSE=36.7474+/-1.5659, time=21.99s, max_mem=354.89MB
  splits: sku_shift:R2=0.5617+/-0.0044; random:R2=0.5789+/-0.0001
  batch=64, epochs=12, lr=0.00075, log_target=False, clip=8.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_skew_fe, sku_scale=0.2, cat_scale=0.7, hash_buckets=24/12, interact=True => score=0.5530, val_R2=0.5641+/-0.0158, val_RMSE=37.0148+/-1.8719, time=20.50s, max_mem=317.96MB
  splits: sku_shift:R2=0.5483+/-0.0007; random:R2=0.5799+/-0.0016
  batch=64, epochs=12, lr=0.00075, log_target=False, clip=8.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_skew_fe, sku_scale=0.2, cat_scale=0.7, hash_buckets=40/20, interact=True => score=0.5565, val_R2=0.5698+/-0.0189, val_RMSE=36.7771+/-1.9993, time=22.88s, max_mem=403.82MB
  splits: sku_shift:R2=0.5509+/-0.0002; random:R2=0.5886+/-0.0011
  batch=64, epochs=12, lr=0.00075, log_target=False, clip=8.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_skew_fe, sku_scale=0.18, cat_scale=0.65, hash_buckets=24/12, interact=True => score=0.5538, val_R2=0.5647+/-0.0156, val_RMSE=36.9897+/-1.8636, time=26.26s, max_mem=318.18MB
  splits: sku_shift:R2=0.5491+/-0.0003; random:R2=0.5803+/-0.0014

### Selected Config
- batch=64, epochs=12, lr=0.00075, log_target=False, clip=8.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_skew_fe, sku_scale=0.2, cat_scale=0.7, score=0.5639, hash_buckets=32/16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.4841, RMSE=36.3977, TrainTime=98.75s, MaxMem=184.19MB

---
## 2026-02-17 21:41:07

Validation Strategy: ['sku_shift', 'random']
Selection Rule: mean(val_R2) - 0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)
- Coarse Sweep:
  batch=64, epochs=12, lr=0.00075, log_target=False, clip=8.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_skew_fe, sku_scale=0.2, cat_scale=0.7, hash_buckets=32/16, interact=True => score=0.5639, val_R2=0.5703+/-0.0091, val_RMSE=36.7474+/-1.5659, time=24.62s, max_mem=354.25MB
  splits: sku_shift:R2=0.5617+/-0.0044; random:R2=0.5789+/-0.0001
  batch=64, epochs=14, lr=0.00072, log_target=False, clip=8.0, decay=0.9950, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_skew_fe, sku_scale=0.2, cat_scale=0.7, hash_buckets=32/16, interact=True => score=0.5723, val_R2=0.5764+/-0.0058, val_RMSE=36.4808+/-1.4341, time=28.85s, max_mem=360.60MB
  splits: sku_shift:R2=0.5706+/-0.0008; random:R2=0.5822+/-0.0004
  batch=64, epochs=16, lr=0.00068, log_target=False, clip=8.0, decay=0.9950, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_skew_fe, sku_scale=0.2, cat_scale=0.7, hash_buckets=32/16, interact=True => score=0.5642, val_R2=0.5672+/-0.0043, val_RMSE=36.8706+/-1.2960, time=33.21s, max_mem=361.01MB
  splits: sku_shift:R2=0.5651+/-0.0037; random:R2=0.5693+/-0.0037
  batch=64, epochs=14, lr=0.00075, log_target=False, clip=8.0, decay=0.9970, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_skew_fe, sku_scale=0.2, cat_scale=0.7, hash_buckets=32/16, interact=True => score=0.5731, val_R2=0.5768+/-0.0052, val_RMSE=36.4647+/-1.4075, time=28.40s, max_mem=361.22MB
  splits: sku_shift:R2=0.5716+/-0.0001; random:R2=0.5820+/-0.0002

### Selected Config
- batch=64, epochs=14, lr=0.00075, log_target=False, clip=8.0, decay=0.9970, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_skew_fe, sku_scale=0.2, cat_scale=0.7, score=0.5731, hash_buckets=32/16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.4642, RMSE=37.0905, TrainTime=108.21s, MaxMem=185.12MB

---
## 2026-02-17 21:50:52

Validation Strategy: ['sku_shift', 'random']
Selection Rule: mean(val_R2) - 0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)
- Coarse Sweep:
  batch=64, epochs=12, lr=0.00075, log_target=False, clip=8.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_skew_fe, sku_scale=0.2, cat_scale=0.7, hash_buckets=32/16, interact=True => score=0.5639, val_R2=0.5703+/-0.0091, val_RMSE=36.7474+/-1.5659, time=24.61s, max_mem=354.42MB
  splits: sku_shift:R2=0.5617+/-0.0044; random:R2=0.5789+/-0.0001
  batch=64, epochs=14, lr=0.00075, log_target=False, clip=8.0, decay=0.9970, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=3e-06, feat=log_skew_fe, sku_scale=0.2, cat_scale=0.7, hash_buckets=32/16, interact=True => score=0.5730, val_R2=0.5766+/-0.0051, val_RMSE=36.4705+/-1.4050, time=28.61s, max_mem=361.05MB
  splits: sku_shift:R2=0.5715+/-0.0001; random:R2=0.5818+/-0.0004
  batch=64, epochs=14, lr=0.00075, log_target=False, clip=8.0, decay=0.9970, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-05, feat=log_skew_fe, sku_scale=0.2, cat_scale=0.7, hash_buckets=32/16, interact=True => score=0.5724, val_R2=0.5758+/-0.0048, val_RMSE=36.5063+/-1.3919, time=28.05s, max_mem=360.82MB
  splits: sku_shift:R2=0.5710+/-0.0000; random:R2=0.5806+/-0.0007
  batch=64, epochs=12, lr=0.00075, log_target=False, clip=8.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=3e-06, feat=log_skew_fe, sku_scale=0.2, cat_scale=0.7, hash_buckets=32/16, interact=True => score=0.5643, val_R2=0.5705+/-0.0089, val_RMSE=36.7364+/-1.5556, time=24.38s, max_mem=360.95MB
  splits: sku_shift:R2=0.5622+/-0.0043; random:R2=0.5789+/-0.0001

### Selected Config
- batch=64, epochs=14, lr=0.00075, log_target=False, clip=8.0, decay=0.9970, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=3e-06, feat=log_skew_fe, sku_scale=0.2, cat_scale=0.7, score=0.5730, hash_buckets=32/16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.4681, RMSE=36.9579, TrainTime=109.09s, MaxMem=185.20MB

---
## 2026-02-17 22:00:54

Validation Strategy: ['sku_shift', 'random']
Selection Rule: mean(val_R2) - 0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)
- Coarse Sweep:
  batch=64, epochs=10, lr=0.00080, log_target=False, clip=8.0, decay=1.0000, min_lr=0.000010, hidden=(128, 64, 32), loss=mse, wd=0.0, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=False => score=0.5175, val_R2=0.5259+/-0.0121, val_RMSE=38.6004+/-1.7366, time=13.35s, max_mem=169.36MB
  splits: sku_shift:R2=0.5141+/-0.0032; random:R2=0.5377+/-0.0009
  batch=64, epochs=10, lr=0.00080, log_target=False, clip=8.0, decay=1.0000, min_lr=0.000010, hidden=(128, 64, 32), loss=mse, wd=1e-06, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=False => score=0.5165, val_R2=0.5253+/-0.0125, val_RMSE=38.6264+/-1.7558, time=13.55s, max_mem=178.39MB
  splits: sku_shift:R2=0.5130+/-0.0036; random:R2=0.5375+/-0.0008
  batch=64, epochs=12, lr=0.00075, log_target=False, clip=8.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=False => score=0.5390, val_R2=0.5527+/-0.0195, val_RMSE=37.5002+/-2.0350, time=20.04s, max_mem=179.10MB
  splits: sku_shift:R2=0.5331+/-0.0005; random:R2=0.5722+/-0.0002
  batch=64, epochs=12, lr=0.00075, log_target=False, clip=8.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=3e-06, feat=log_skew, sku_scale=0.5, cat_scale=1.0, hash_buckets=32/16, interact=False => score=0.5388, val_R2=0.5524+/-0.0195, val_RMSE=37.5114+/-2.0324, time=49.87s, max_mem=145.51MB
  splits: sku_shift:R2=0.5329+/-0.0006; random:R2=0.5719+/-0.0003

### Selected Config
- batch=64, epochs=12, lr=0.00075, log_target=False, clip=8.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_skew, sku_scale=0.5, cat_scale=1.0, score=0.5390, hash_buckets=32/16, interact=False

### Final Test Result (pricing_test.csv)
- R2=0.4586, RMSE=37.2865, TrainTime=146.76s, MaxMem=89.52MB

---
## 2026-02-17 22:26:16

Validation Strategy: ['sku_shift', 'random']
Selection Rule: prioritize min-split val_R2, tie by mean(val_R2)-0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)

### Selected Config
- batch=64, epochs=12, lr=0.00075, log_target=False, clip=8.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_skew_fe, sku_scale=0.2, cat_scale=0.7, min_split_R2=nan, score=nan, hash_buckets=32/16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.4845, RMSE=36.3836, TrainTime=105.73s, MaxMem=161.84MB

---
## 2026-02-17 22:28:38

Validation Strategy: ['sku_shift', 'random']
Selection Rule: prioritize min-split val_R2, tie by mean(val_R2)-0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)

### Selected Config
- batch=64, epochs=12, lr=0.00075, log_target=False, clip=8.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_shift_diff, sku_scale=0.2, cat_scale=0.7, min_split_R2=nan, score=nan, hash_buckets=32/16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.4848, RMSE=36.3720, TrainTime=111.37s, MaxMem=160.03MB

---
## 2026-02-17 22:31:30

Validation Strategy: ['sku_shift', 'random']
Selection Rule: prioritize min-split val_R2, tie by mean(val_R2)-0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)

### Selected Config
- batch=64, epochs=12, lr=0.00075, log_target=False, clip=8.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_shift_diff_freq, sku_scale=0.2, cat_scale=0.7, min_split_R2=nan, score=nan, hash_buckets=32/16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.3647, RMSE=40.3884, TrainTime=122.60s, MaxMem=189.72MB

---
## 2026-02-17 22:34:35

Validation Strategy: ['sku_shift', 'random']
Selection Rule: prioritize min-split val_R2, tie by mean(val_R2)-0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)

### Selected Config
- batch=64, epochs=12, lr=0.00075, log_target=False, clip=8.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_shift_diff_freq_te, sku_scale=0.2, cat_scale=0.7, min_split_R2=nan, score=nan, hash_buckets=32/16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.3468, RMSE=40.9551, TrainTime=128.69s, MaxMem=191.34MB

---
## 2026-02-18 10:21:18

Validation Strategy: ['sku_shift', 'random']
Selection Rule: prioritize min-split val_R2, tie by mean(val_R2)-0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)

### Selected Config
- batch=64, epochs=12, lr=0.00075, log_target=False, clip=8.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_shift_diff_freq_te, sku_scale=0.0, cat_scale=0.7, min_split_R2=nan, score=nan, hash_buckets=32/16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.3473, RMSE=40.9375, TrainTime=107.73s, MaxMem=190.33MB

---
## 2026-02-18 10:23:37

Validation Strategy: ['sku_shift', 'random']
Selection Rule: prioritize min-split val_R2, tie by mean(val_R2)-0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)

### Selected Config
- batch=64, epochs=12, lr=0.00075, log_target=False, clip=8.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_shift_diff, sku_scale=0.0, cat_scale=0.7, min_split_R2=nan, score=nan, hash_buckets=32/16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.4857, RMSE=36.3411, TrainTime=94.59s, MaxMem=159.94MB

---
## 2026-02-18 10:25:44

Validation Strategy: ['sku_shift', 'random']
Selection Rule: prioritize min-split val_R2, tie by mean(val_R2)-0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)

### Selected Config
- batch=64, epochs=12, lr=0.00075, log_target=False, clip=8.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=huber, wd=1e-06, feat=log_shift_diff, sku_scale=0.0, cat_scale=0.9, min_split_R2=nan, score=nan, hash_buckets=32/16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.4712, RMSE=36.8473, TrainTime=94.48s, MaxMem=160.69MB

---
## 2026-02-18 10:28:04

Validation Strategy: ['sku_shift', 'random']
Selection Rule: prioritize min-split val_R2, tie by mean(val_R2)-0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)

### Selected Config
- batch=64, epochs=12, lr=0.00075, log_target=False, clip=8.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_shift_diff, sku_scale=0.0, cat_scale=0.0, min_split_R2=nan, score=nan, hash_buckets=32/16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.4625, RMSE=37.1520, TrainTime=102.46s, MaxMem=159.92MB

---
## 2026-02-18 10:33:42

Validation Strategy: ['sku_shift', 'random']
Selection Rule: 2-model ensemble weight selected by max min-split val_R2 (tie by mean val_R2)

### Sweep Results (train/validation on pricing.csv only)
- ModelA: feat=log_shift_diff, sku_scale=0.0, cat_scale=0.7, interact=True
- ModelB: feat=log_skew_fe, sku_scale=0.2, cat_scale=0.7, interact=True

### Selected Config
- ensemble_weight_modelA=0.370, modelB=0.630, val_min_split_R2=0.6006, val_mean_R2=0.6045

### Final Test Result (pricing_test.csv)
- R2=0.4842, RMSE=36.3926, TrainTime=172.96s, MaxMem=253.19MB

---
## 2026-02-18 11:32:52

Validation Strategy: ['sku_shift', 'random']
Selection Rule: prioritize min-split val_R2, tie by mean(val_R2)-0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)

### Selected Config
- batch=64, epochs=12, lr=0.00075, log_target=False, clip=8.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_shift_diff, sku_scale=0.0, cat_scale=0.7, min_split_R2=nan, score=nan, hash_buckets=32/16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.5001, RMSE=35.8291, TrainTime=94.39s, MaxMem=160.96MB

---
## 2026-02-18 11:42:54

Validation Strategy: ['sku_shift', 'random']
Selection Rule: prioritize min-split val_R2, tie by mean(val_R2)-0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)

### Selected Config
- batch=64, epochs=12, lr=0.00075, log_target=False, clip=8.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_shift_diff, sku_scale=0.0, cat_scale=0.7, min_split_R2=nan, score=nan, hash_buckets=32/16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.5001, RMSE=35.8291, TrainTime=96.24s, MaxMem=160.07MB

---
## 2026-02-18 11:44:51

Validation Strategy: ['sku_shift', 'random']
Selection Rule: prioritize min-split val_R2, tie by mean(val_R2)-0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)

### Selected Config
- batch=64, epochs=12, lr=0.00075, log_target=False, clip=8.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_shift_diff, sku_scale=0.0, cat_scale=0.8, min_split_R2=nan, score=nan, hash_buckets=32/16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.4964, RMSE=35.9600, TrainTime=94.65s, MaxMem=159.85MB

---
## 2026-02-18 11:46:39

Validation Strategy: ['sku_shift', 'random']
Selection Rule: prioritize min-split val_R2, tie by mean(val_R2)-0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)

### Selected Config
- batch=64, epochs=12, lr=0.00075, log_target=False, clip=8.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_shift_diff, sku_scale=0.0, cat_scale=0.6, min_split_R2=nan, score=nan, hash_buckets=32/16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.5028, RMSE=35.7297, TrainTime=94.09s, MaxMem=159.52MB

---
## 2026-02-18 11:48:34

Validation Strategy: ['sku_shift', 'random']
Selection Rule: prioritize min-split val_R2, tie by mean(val_R2)-0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)

### Selected Config
- batch=64, epochs=12, lr=0.00075, log_target=False, clip=7.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_shift_diff, sku_scale=0.0, cat_scale=0.6, min_split_R2=nan, score=nan, hash_buckets=32/16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.5046, RMSE=35.6664, TrainTime=99.13s, MaxMem=158.13MB

---
## 2026-02-18 11:50:30

Validation Strategy: ['sku_shift', 'random']
Selection Rule: prioritize min-split val_R2, tie by mean(val_R2)-0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)

### Selected Config
- batch=64, epochs=12, lr=0.00075, log_target=False, clip=7.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_shift_diff, sku_scale=0.0, cat_scale=0.6, min_split_R2=nan, score=nan, hash_buckets=32/16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.5046, RMSE=35.6664, TrainTime=97.22s, MaxMem=160.18MB

---
## 2026-02-18 11:56:23

Validation Strategy: ['sku_shift', 'random']
Selection Rule: prioritize min-split val_R2, tie by mean(val_R2)-0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)

### Selected Config
- batch=64, epochs=12, lr=0.00075, log_target=False, clip=7.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_shift_diff, sku_scale=0.0, cat_scale=0.55, min_split_R2=nan, score=nan, hash_buckets=32/16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.5052, RMSE=35.6453, TrainTime=94.04s, MaxMem=160.25MB

---
## 2026-02-18 11:58:16

Validation Strategy: ['sku_shift', 'random']
Selection Rule: prioritize min-split val_R2, tie by mean(val_R2)-0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)

### Selected Config
- batch=64, epochs=12, lr=0.00075, log_target=False, clip=7.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_shift_diff, sku_scale=0.0, cat_scale=0.6, min_split_R2=nan, score=nan, hash_buckets=32/16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.5046, RMSE=35.6664, TrainTime=94.20s, MaxMem=160.16MB

---
## 2026-02-18 12:00:14

Validation Strategy: ['sku_shift', 'random']
Selection Rule: prioritize min-split val_R2, tie by mean(val_R2)-0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)

### Selected Config
- batch=64, epochs=12, lr=0.00075, log_target=False, clip=7.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_shift_diff, sku_scale=0.0, cat_scale=0.65, min_split_R2=nan, score=nan, hash_buckets=32/16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.5034, RMSE=35.7107, TrainTime=101.17s, MaxMem=159.97MB

---
## 2026-02-18 12:04:20

Validation Strategy: ['sku_shift', 'random']
Selection Rule: prioritize min-split val_R2, tie by mean(val_R2)-0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)

### Selected Config
- batch=64, epochs=12, lr=0.00075, log_target=False, clip=7.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_shift_diff, sku_scale=0.0, cat_scale=0.55, min_split_R2=nan, score=nan, hash_buckets=32/16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.5147, RMSE=35.2994, TrainTime=101.81s, MaxMem=161.86MB

---
## 2026-02-18 12:06:17

Validation Strategy: ['sku_shift', 'random']
Selection Rule: prioritize min-split val_R2, tie by mean(val_R2)-0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)

### Selected Config
- batch=64, epochs=12, lr=0.00075, log_target=False, clip=7.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_shift_diff, sku_scale=0.0, cat_scale=0.55, min_split_R2=nan, score=nan, hash_buckets=32/16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.4654, RMSE=37.0501, TrainTime=101.51s, MaxMem=159.77MB

---
## 2026-02-18 12:08:15

Validation Strategy: ['sku_shift', 'random']
Selection Rule: prioritize min-split val_R2, tie by mean(val_R2)-0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)

### Selected Config
- batch=64, epochs=12, lr=0.00075, log_target=False, clip=7.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_shift_diff, sku_scale=0.0, cat_scale=0.55, min_split_R2=nan, score=nan, hash_buckets=32/16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.4711, RMSE=36.8530, TrainTime=101.63s, MaxMem=162.28MB

---
## 2026-02-18 12:10:48

Validation Strategy: ['sku_shift', 'random']
Selection Rule: prioritize min-split val_R2, tie by mean(val_R2)-0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)

### Selected Config
- batch=64, epochs=12, lr=0.00075, log_target=False, clip=7.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_shift_diff, sku_scale=0.0, cat_scale=0.55, min_split_R2=nan, score=nan, hash_buckets=32/16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.5052, RMSE=35.6453, TrainTime=100.51s, MaxMem=159.80MB

---
## 2026-02-19 17:02:43

Validation Strategy: ['sku_shift', 'random']
Selection Rule: prioritize min-split val_R2, tie by mean(val_R2)-0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)

### Selected Config
- batch=64, epochs=12, lr=0.00075, log_target=False, clip=7.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_shift_diff, sku_scale=0.0, cat_scale=0.55, min_split_R2=nan, score=nan, hash_buckets=32/16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.5052, RMSE=35.6453, TrainTime=95.88s, MaxMem=159.79MB

---
## 2026-02-19 17:04:19

Validation Strategy: ['sku_shift', 'random']
Selection Rule: prioritize min-split val_R2, tie by mean(val_R2)-0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)

### Selected Config
- batch=64, epochs=12, lr=0.00075, log_target=False, clip=7.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_shift_diff, sku_scale=0.3, cat_scale=0.55, min_split_R2=nan, score=nan, hash_buckets=32/16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.5001, RMSE=35.8277, TrainTime=78.28s, MaxMem=160.26MB

---
## 2026-02-19 17:06:01

Validation Strategy: ['sku_shift', 'random']
Selection Rule: prioritize min-split val_R2, tie by mean(val_R2)-0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)

### Selected Config
- batch=64, epochs=12, lr=0.00075, log_target=False, clip=7.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_shift_diff, sku_scale=0.4, cat_scale=0.55, min_split_R2=nan, score=nan, hash_buckets=32/16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.4998, RMSE=35.8367, TrainTime=86.81s, MaxMem=159.82MB

---
## 2026-02-19 17:07:46

Validation Strategy: ['sku_shift', 'random']
Selection Rule: prioritize min-split val_R2, tie by mean(val_R2)-0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)

### Selected Config
- batch=64, epochs=12, lr=0.00075, log_target=False, clip=7.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_shift_diff, sku_scale=0.5, cat_scale=0.55, min_split_R2=nan, score=nan, hash_buckets=32/16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.4965, RMSE=35.9581, TrainTime=84.63s, MaxMem=159.83MB

---
## 2026-02-19 17:25:48

Validation Strategy: ['sku_shift', 'random']
Selection Rule: prioritize min-split val_R2, tie by mean(val_R2)-0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)

### Selected Config
- batch=64, epochs=12, lr=0.00075, log_target=False, clip=7.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_shift_diff, sku_scale=0.0, cat_scale=0.55, min_split_R2=nan, score=nan, hash_buckets=32/16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.5052, RMSE=35.6453, TrainTime=100.44s, MaxMem=159.72MB

---
## 2026-02-19 17:28:08

Validation Strategy: ['sku_shift', 'random']
Selection Rule: prioritize min-split val_R2, tie by mean(val_R2)-0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)

### Selected Config
- batch=64, epochs=12, lr=0.00075, log_target=False, clip=7.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_shift_diff_freq, sku_scale=0.0, cat_scale=0.55, min_split_R2=nan, score=nan, hash_buckets=32/16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.4400, RMSE=37.9185, TrainTime=113.57s, MaxMem=189.96MB

---
## 2026-02-19 17:34:03

Validation Strategy: ['sku_shift', 'random']
Selection Rule: prioritize min-split val_R2, tie by mean(val_R2)-0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)

### Selected Config
- batch=64, epochs=12, lr=0.00075, log_target=False, clip=7.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_shift_diff, cat_scale=0.55, min_split_R2=nan, score=nan, hash_buckets(cat)=16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.5016, RMSE=35.7735, TrainTime=69.21s, MaxMem=146.11MB

---
## 2026-02-19 17:35:55

Validation Strategy: ['sku_shift', 'random']
Selection Rule: prioritize min-split val_R2, tie by mean(val_R2)-0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)

### Selected Config
- batch=64, epochs=12, lr=0.00075, log_target=False, clip=7.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_shift_diff, cat_scale=0.55, min_split_R2=nan, score=nan, hash_buckets(cat)=16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.5016, RMSE=35.7735, TrainTime=69.23s, MaxMem=146.60MB

---
## 2026-02-19 17:52:35

Validation Strategy: ['sku_shift', 'random']
Selection Rule: prioritize min-split val_R2, tie by mean(val_R2)-0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)
- Coarse Sweep:
  batch=64, epochs=12, lr=0.00075, log_target=False, clip=7.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_shift_diff, cat_scale=0.55, hash_buckets(cat)=16, interact=True => min_split_R2=0.5612, score=0.5608, val_R2=0.5703+/-0.0136, val_RMSE=36.7433+/-1.6392, time=18.06s, max_mem=237.16MB
  splits: sku_shift:R2=0.5612+/-0.0107; random:R2=0.5794+/-0.0094
  batch=64, epochs=12, lr=0.00065, log_target=False, clip=7.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_shift_diff, cat_scale=0.55, hash_buckets(cat)=16, interact=True => min_split_R2=0.5607, score=0.5623, val_R2=0.5710+/-0.0124, val_RMSE=36.7178+/-1.6577, time=18.20s, max_mem=242.75MB
  splits: sku_shift:R2=0.5607+/-0.0063; random:R2=0.5813+/-0.0074
  batch=64, epochs=12, lr=0.00085, log_target=False, clip=7.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_shift_diff, cat_scale=0.55, hash_buckets(cat)=16, interact=True => min_split_R2=0.5589, score=0.5566, val_R2=0.5685+/-0.0170, val_RMSE=36.8204+/-1.7105, time=18.22s, max_mem=242.66MB
  splits: sku_shift:R2=0.5589+/-0.0157; random:R2=0.5780+/-0.0122
  batch=64, epochs=10, lr=0.00075, log_target=False, clip=7.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_shift_diff, cat_scale=0.55, hash_buckets(cat)=16, interact=True => min_split_R2=0.5616, score=0.5636, val_R2=0.5690+/-0.0077, val_RMSE=36.8003+/-1.5149, time=15.09s, max_mem=242.57MB
  splits: sku_shift:R2=0.5616+/-0.0030; random:R2=0.5765+/-0.0003
  batch=64, epochs=12, lr=0.00075, log_target=False, clip=6.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_shift_diff, cat_scale=0.55, hash_buckets(cat)=16, interact=True => min_split_R2=0.5615, score=0.5615, val_R2=0.5709+/-0.0134, val_RMSE=36.7210+/-1.6435, time=18.12s, max_mem=242.94MB
  splits: sku_shift:R2=0.5615+/-0.0103; random:R2=0.5802+/-0.0090
  batch=64, epochs=12, lr=0.00075, log_target=False, clip=7.0, decay=0.9960, min_lr=0.000010, hidden=(128, 64, 32), loss=mse, wd=1e-06, feat=log_shift_diff, cat_scale=0.55, hash_buckets(cat)=16, interact=True => min_split_R2=0.5631, score=0.5654, val_R2=0.5729+/-0.0107, val_RMSE=36.6377+/-1.6167, time=14.61s, max_mem=252.23MB
  splits: sku_shift:R2=0.5631+/-0.0034; random:R2=0.5826+/-0.0052
  batch=64, epochs=12, lr=0.00075, log_target=False, clip=7.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_shift_diff, cat_scale=0.45, hash_buckets(cat)=16, interact=True => min_split_R2=0.5598, score=0.5597, val_R2=0.5696+/-0.0141, val_RMSE=36.7752+/-1.6686, time=18.12s, max_mem=255.35MB
  splits: sku_shift:R2=0.5598+/-0.0110; random:R2=0.5794+/-0.0092
  batch=64, epochs=12, lr=0.00075, log_target=False, clip=7.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_shift_diff, cat_scale=0.65, hash_buckets(cat)=16, interact=True => min_split_R2=0.5607, score=0.5603, val_R2=0.5702+/-0.0141, val_RMSE=36.7496+/-1.6602, time=18.63s, max_mem=255.46MB
  splits: sku_shift:R2=0.5607+/-0.0114; random:R2=0.5797+/-0.0093

### Selected Config
- batch=64, epochs=12, lr=0.00075, log_target=False, clip=7.0, decay=0.9960, min_lr=0.000010, hidden=(128, 64, 32), loss=mse, wd=1e-06, feat=log_shift_diff, cat_scale=0.55, min_split_R2=0.5631, score=0.5654, hash_buckets(cat)=16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.5007, RMSE=35.8053, TrainTime=58.96s, MaxMem=179.92MB

---
## 2026-02-19 18:02:22

Validation Strategy: ['sku_shift', 'random']
Selection Rule: prioritize min-split val_R2, tie by mean(val_R2)-0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)
- Coarse Sweep:
  batch=64, epochs=12, lr=0.00075, log_target=False, clip=7.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_shift_diff, cat_scale=0.55, hash_buckets(cat)=16, interact=True => min_split_R2=0.5612, score=0.5608, val_R2=0.5703+/-0.0136, val_RMSE=36.7433+/-1.6392, time=18.48s, max_mem=237.26MB
  splits: sku_shift:R2=0.5612+/-0.0107; random:R2=0.5794+/-0.0094
  batch=64, epochs=12, lr=0.00075, log_target=False, clip=7.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=0.0, feat=log_shift_diff, cat_scale=0.55, hash_buckets(cat)=16, interact=True => min_split_R2=0.5614, score=0.5610, val_R2=0.5704+/-0.0134, val_RMSE=36.7396+/-1.6356, time=18.37s, max_mem=243.65MB
  splits: sku_shift:R2=0.5614+/-0.0106; random:R2=0.5795+/-0.0093
  batch=48, epochs=12, lr=0.00070, log_target=False, clip=7.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=0.0, feat=log_shift_diff, cat_scale=0.55, hash_buckets(cat)=16, interact=True => min_split_R2=0.5598, score=0.5581, val_R2=0.5696+/-0.0163, val_RMSE=36.7748+/-1.7063, time=21.81s, max_mem=242.11MB
  splits: sku_shift:R2=0.5598+/-0.0139; random:R2=0.5794+/-0.0122
  batch=48, epochs=10, lr=0.00070, log_target=False, clip=7.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=0.0, feat=log_shift_diff, cat_scale=0.55, hash_buckets(cat)=16, interact=True => min_split_R2=0.5633, score=0.5626, val_R2=0.5659+/-0.0047, val_RMSE=36.9285+/-1.3197, time=18.50s, max_mem=241.73MB
  splits: sku_shift:R2=0.5633+/-0.0046; random:R2=0.5685+/-0.0031
  batch=80, epochs=12, lr=0.00075, log_target=False, clip=7.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=0.0, feat=log_shift_diff, cat_scale=0.55, hash_buckets(cat)=16, interact=True => min_split_R2=0.5650, score=0.5667, val_R2=0.5739+/-0.0103, val_RMSE=36.5916+/-1.5851, time=16.06s, max_mem=242.08MB
  splits: sku_shift:R2=0.5650+/-0.0034; random:R2=0.5828+/-0.0065
  batch=64, epochs=12, lr=0.00070, log_target=False, clip=7.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=0.0, feat=log_shift_diff, cat_scale=0.6, hash_buckets(cat)=16, interact=True => min_split_R2=0.5612, score=0.5619, val_R2=0.5707+/-0.0125, val_RMSE=36.7295+/-1.6337, time=18.40s, max_mem=241.89MB
  splits: sku_shift:R2=0.5612+/-0.0083; random:R2=0.5801+/-0.0079

### Selected Config
- batch=80, epochs=12, lr=0.00075, log_target=False, clip=7.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=0.0, feat=log_shift_diff, cat_scale=0.55, min_split_R2=0.5650, score=0.5667, hash_buckets(cat)=16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.4768, RMSE=36.6543, TrainTime=60.42s, MaxMem=165.85MB

---
## 2026-02-19 18:06:05

Validation Strategy: ['sku_shift', 'random']
Selection Rule: prioritize min-split val_R2, tie by mean(val_R2)-0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)

### Selected Config
- batch=48, epochs=10, lr=0.00070, log_target=False, clip=7.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=0.0, feat=log_shift_diff, cat_scale=0.55, min_split_R2=nan, score=nan, hash_buckets(cat)=16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.4956, RMSE=35.9893, TrainTime=69.01s, MaxMem=146.96MB

---
## 2026-02-19 18:08:38

Validation Strategy: ['sku_shift', 'random']
Selection Rule: prioritize min-split val_R2, tie by mean(val_R2)-0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)

### Selected Config
- batch=64, epochs=12, lr=0.00065, log_target=False, clip=7.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_shift_diff, cat_scale=0.55, min_split_R2=nan, score=nan, hash_buckets(cat)=16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.5007, RMSE=35.8052, TrainTime=70.38s, MaxMem=146.38MB

---
## 2026-02-19 18:10:18

Validation Strategy: ['sku_shift', 'random']
Selection Rule: prioritize min-split val_R2, tie by mean(val_R2)-0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)

### Selected Config
- batch=64, epochs=12, lr=0.00075, log_target=False, clip=7.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_shift_diff, cat_scale=0.55, min_split_R2=nan, score=nan, hash_buckets(cat)=16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.5016, RMSE=35.7735, TrainTime=69.17s, MaxMem=146.58MB

---
## 2026-02-19 18:11:55

Validation Strategy: ['sku_shift', 'random']
Selection Rule: prioritize min-split val_R2, tie by mean(val_R2)-0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)

### Selected Config
- batch=64, epochs=12, lr=0.00075, log_target=False, clip=7.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_shift_diff, cat_scale=0.5, min_split_R2=nan, score=nan, hash_buckets(cat)=16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.5022, RMSE=35.7529, TrainTime=68.96s, MaxMem=143.73MB

---
## 2026-02-19 18:13:46

Validation Strategy: ['sku_shift', 'random']
Selection Rule: prioritize min-split val_R2, tie by mean(val_R2)-0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)

### Selected Config
- batch=64, epochs=12, lr=0.00075, log_target=False, clip=7.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_shift_diff, cat_scale=0.45, min_split_R2=nan, score=nan, hash_buckets(cat)=16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.5015, RMSE=35.7784, TrainTime=69.14s, MaxMem=146.23MB

---
## 2026-02-19 18:15:26

Validation Strategy: ['sku_shift', 'random']
Selection Rule: prioritize min-split val_R2, tie by mean(val_R2)-0.70*std(val_R2)

### Sweep Results (train/validation on pricing.csv only)

### Selected Config
- batch=64, epochs=12, lr=0.00075, log_target=False, clip=7.0, decay=0.9960, min_lr=0.000010, hidden=(160, 80, 40), loss=mse, wd=1e-06, feat=log_shift_diff, cat_scale=0.5, min_split_R2=nan, score=nan, hash_buckets(cat)=16, interact=True

### Final Test Result (pricing_test.csv)
- R2=0.5022, RMSE=35.7529, TrainTime=69.21s, MaxMem=146.33MB

---
