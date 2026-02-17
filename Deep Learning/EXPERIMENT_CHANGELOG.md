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
