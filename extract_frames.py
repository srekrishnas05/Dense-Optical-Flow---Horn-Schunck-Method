import cv2
import os

VIDEO_PATH = r"C:\Users\sreks\Downloads\imgprocfp\imgprocvid.mp4"
OUT_DIR = r"C:\Users\sreks\Downloads\imgprocfp"


FRAME0 = 0
DELTA = 1  

os.makedirs(OUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

def grab(idx: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Could not read frame {idx}")
    return frame

f1 = grab(FRAME0)
f2 = grab(FRAME0 + DELTA)

cv2.imwrite(os.path.join(OUT_DIR, "frame1.png"), f1)
cv2.imwrite(os.path.join(OUT_DIR, "frame2.png"), f2)

print(f"Saved frames: {OUT_DIR}/frame1.png and {OUT_DIR}/frame2.png")
cap.release()
