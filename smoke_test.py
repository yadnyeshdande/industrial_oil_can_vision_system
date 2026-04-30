import sys
import traceback

def main():
    try:
        import numpy as np
        import cv2
        import torch
        from ultralytics import YOLO

        print('numpy', np.__version__)
        print('cv2', cv2.__version__)
        print('torch', torch.__version__, 'cuda_available=', torch.cuda.is_available())

        # quick model load on CPU to validate ultralytics integration
        m = YOLO('models/best.pt')
        print('YOLO model loaded')

        # create a single dummy image and run one prediction
        dummy = (np.zeros((720, 1280, 3), dtype=np.uint8) + 127)
        res = m.predict(source=[dummy], device='cpu', stream=False, conf=0.25)
        print('Prediction completed, results length:', len(res))
        print('SMOKE TEST: SUCCESS')
        return 0
    except Exception:
        traceback.print_exc()
        print('SMOKE TEST: FAILED', file=sys.stderr)
        return 2

if __name__ == '__main__':
    raise SystemExit(main())
