import cv2
import socket
import time
import sys

def test_connectivity(ip, port=554):
    """Test if the camera is reachable"""
    try:
        sock = socket.create_connection((ip, port), timeout=2)
        sock.close()
        print(f"✓ Camera is reachable on {ip}:{port}")
        return True
    except (socket.timeout, ConnectionRefusedError, OSError) as e:
        print(f"✗ Cannot reach {ip}:{port} - {e}")
        return False

def test_http_access(ip, port=80):
    """Test HTTP access to camera web interface"""
    try:
        sock = socket.create_connection((ip, port), timeout=2)
        sock.close()
        print(f"✓ HTTP access available on {ip}:{port}")
        print(f"  Try: http://{ip}:{port} in your browser")
        return True
    except:
        return False

def test_rtsp_url(url, timeout=5):
    """Test a single RTSP URL"""
    print(f"  Testing: {url}")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    start = time.time()
    while time.time() - start < timeout:
        if cap.grab():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"    ✓ CONNECTED! Frame: {frame.shape}")
                cap.release()
                return True
        time.sleep(0.05)

    cap.release()
    return False

# Camera info
ip = "192.168.1.249"
username = "admin"
password = "Pass_123"

print("="*70)
print("CP Plus Camera Discovery Tool")
print("="*70)
print(f"Target IP: {ip}\n")

# Step 1: Test connectivity
print("[1] Testing basic connectivity...")
if not test_connectivity(ip, 554):
    print("\n[!] Cannot reach camera. Possible issues:")
    print("    - Camera IP is wrong")
    print("    - Camera is offline")
    print("    - Network unreachable")
    print("    - Firewall blocking")
    test_http_access(ip, 80)
    sys.exit(1)

print()

# Step 2: Try various RTSP paths
print("[2] Trying common CP Plus RTSP paths...")
print()

base = f"rtsp://{username}:{password}@{ip}:554"
base_alt = f"rtsp://{username}:{password}@{ip}:8554"

urls_to_try = [
    # Standard CP Plus paths
    f"{base}/cam/realmonitor?channel=1&subtype=0",
    f"{base}/cam/realmonitor?channel=1",
    f"{base}/Streaming/Channels/101",
    f"{base}/Streaming/Channels/102",
    f"{base}/stream1",
    f"{base}/stream",
    f"{base}/live",
    f"{base}/h264",
    f"{base}/mpeg4",
    f"{base}/video0",
    # Other common variants
    f"{base}/ch0/main/av_stream",
    f"{base}/ch0/sub/av_stream",
    f"{base}/realmonitor?channel=1",
    # Alternative ports
    f"{base_alt}/cam/realmonitor?channel=1&subtype=0",
    f"{base_alt}/stream1",
    # Without auth (if disabled)
    f"rtsp://{ip}:554/cam/realmonitor?channel=1&subtype=0",
    f"rtsp://{ip}:554/stream1",
]

print(f"Testing {len(urls_to_try)} URL candidates:\n")
found = False
for i, url in enumerate(urls_to_try, 1):
    if test_rtsp_url(url, timeout=3):
        print(f"\n✓✓✓ SUCCESS! Working URL: {url}\n")
        found = True
        break
    print()

if not found:
    print("✗ No working URL found in common paths.")
    print("\n[3] Troubleshooting suggestions:")
    print("    1. Check camera web interface:")
    print(f"       http://{ip} or http://{ip}:8000")
    print("    2. Look for RTSP/Stream settings")
    print("    3. Verify RTSP is enabled in camera settings")
    print("    4. Check default credentials (might not be admin/Pass_123)")
    print("    5. Try factory reset if configuration is lost")
    print("\n[4] Manual testing:")
    print('    python camera_discovery.py "rtsp://admin:Pass_123@192.168.1.248:554/YOUR_PATH"')
else:
    print("[!] Save this URL and use it in your camera process!")

# Allow manual URL testing
if len(sys.argv) > 1:
    print("\n" + "="*70)
    print("Manual URL test:")
    manual_url = sys.argv[1]
    print(f"Testing: {manual_url}\n")
    if test_rtsp_url(manual_url, timeout=8):
        print("\n✓✓✓ This URL works!")
    else:
        print("\n✗ This URL does not work")
