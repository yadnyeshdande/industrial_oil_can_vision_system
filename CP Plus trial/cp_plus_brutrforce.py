import cv2

# Replace with your camera's actual information
username = "admin"
password = "Pass_123"
camera_ip = "192.168.1.249" # Replace with your camera's IP address
rtsp_port = 554
stream_path = "cam/realmonitor?channel=1&subtype=0" # Example path for main stream

# Construct the full RTSP URL
rtsp_url = f"rtsp://{username}:{password}@{camera_ip}:{rtsp_port}/"

# Open the video stream using OpenCV's VideoCapture
# You might specify a backend like cv2.CAP_FFMPEG for better compatibility
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print(f"Error: Could not open video stream using {rtsp_url}")
    exit()

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame. Reconnecting or stream ended.")
        # Optional: Add reconnection logic here
        break

    # Display the resulting frame
    cv2.imshow("CP Plus IP Camera Feed", frame)

    # Press 'q' to exit the loop and close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
