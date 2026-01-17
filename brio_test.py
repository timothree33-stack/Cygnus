#!/usr/bin/env python3
import cv2

print("Searching for cameras...")
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            print(f"Camera {i}: {w}x{h} - WORKING")
        cap.release()

# Test camera 0
print("\nTesting camera 0...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

ret, frame = cap.read()
if ret:
    h, w = frame.shape[:2]
    print(f"✅ Brio working: {w}x{h}")
    cv2.imwrite("brio_test.jpg", frame)
    print("Saved: brio_test.jpg")
else:
    print("❌ Failed")
cap.release()
