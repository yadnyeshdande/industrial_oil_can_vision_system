"""
CP Plus Camera - Web Interface Access Guide

Instructions:
1. Open your browser
2. Go to: http://192.168.1.249
3. Login with admin / Pass_123
4. Navigate to:
   - Settings > Network > RTSP
   - OR Streaming > Stream Settings
   - OR Live View > Stream Configuration
5. Look for the RTSP URL or stream path
6. Note the path shown (e.g., "/live", "/stream1", "/cam/realmonitor?channel=1")
7. Come back and test with python camera_discovery.py "rtsp://admin:Pass_123@192.168.1.249:554/YOUR_PATH"

Common locations in CP Plus web interface:
- Settings → Stream Configuration
- Video → Stream
- Network → RTSP
- Live View → Properties

The RTSP URL will look like one of:
  rtsp://admin:Pass_123@192.168.1.249:554/STREAM_PATH
"""

print(__doc__)

# Alternative: Try a few more specific paths
import cv2
import time

def test_url(url):
    print(f"Testing: {url}")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    start = time.time()
    while time.time() - start < 2:
        if cap.grab():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"  ✓ SUCCESS! {frame.shape}")
                cap.release()
                return True
        time.sleep(0.05)

    cap.release()
    return False

# Try a few more uncommon paths
print("\nTesting some additional paths...")
additional_paths = [
    "rtsp://admin:Pass_123@192.168.1.249:554/",
    "rtsp://admin:Pass_123@192.168.1.249:554/profile0",
    "rtsp://admin:Pass_123@192.168.1.249:554/profile1",
    "rtsp://admin:Pass_123@192.168.1.249:554/stream0",
    "rtsp://admin:Pass_123@192.168.1.249:554/stream2",
]

for url in additional_paths:
    test_url(url)
