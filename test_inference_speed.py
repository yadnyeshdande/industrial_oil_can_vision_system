# test_inference_speed.py — run in your venv
import torch, numpy as np, time
from ultralytics import YOLO

model = YOLO("models/best.pt").to("cuda")
dummy = np.zeros((720, 1280, 3), dtype=np.uint8)

# warmup
model.predict(dummy, verbose=False)

times = []
for _ in range(20):
    t = time.perf_counter()
    model.predict(dummy, conf=0.5, verbose=False)
    times.append((time.perf_counter() - t) * 1000)

print(f"Median inference: {sorted(times)[10]:.1f}ms")
print(f"Theoretical max FPS (2 cams): {1000 / (sorted(times)[10] * 2):.1f}")
print(f"Theoretical max FPS (3 cams): {1000 / (sorted(times)[10] * 3):.1f}")