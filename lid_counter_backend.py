import cv2
from ultralytics import YOLO
import threading
import time
from collections import deque

# Anirudh Yuvaraj, Jonathan Philip
# All-CLAD Project, 2026

# Load the custom-trained 99% mAP model
model = YOLO('lidDetection.pt')

# --- Logic Configuration ---
total_inventory = 0
BUFFER_SIZE = 15      # Frames of stability required
PICK_CONFIDENCE = 10  # Frames a lid must be missing while hand-held to count as -1
STACK_SIZE = 5        # Standard box size

# --- State Tracking ---
lid_memory = deque(maxlen=BUFFER_SIZE)
hand_touching_active = False
calibrated = False
missing_count = 0

is_paused = False

def get_centroid(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def is_overlapping(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    # Add a 10% "padding" to the overlap to make it more forgiving for fast hands
    return not (x2 < x3 or x1 > x4 or y2 < y3 or y1 > y4)

class VideoStream:
    def __init__(self, url):
        self.capture = cv2.VideoCapture(url)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.capture.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            self.ret, self.frame = self.capture.read()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.capture.release()

# Initialize Camera
vs = VideoStream("http://192.168.0.52:4747/video").start()
time.sleep(2.0)

while True:
    if not is_paused:
        ret, frame = vs.read()
        if not ret or frame is None: break

        start_time = time.time()
        # High-res inference for precision
        results = model(frame, conf=0.5, imgsz=640, verbose=False)
        inference_speed = (time.time() - start_time) * 1000

        hands = []
        lids = []

        for r in results:
            for box in r.boxes:
                coords = box.xyxy[0].tolist()
                label = model.names[int(box.cls[0])]
                if label == 'hand': hands.append(coords)
                else: lids.append(coords)
                
                # Visual Feedback
                x1, y1, x2, y2 = map(int, coords)
                color = (255, 191, 0) if label == 'hand' else (0, 255, 127) # Modern UI colors
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                cv2.putText(frame, label.upper(), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # --- PRODUCTION LOGIC ENGINE ---
        current_visible = len(lids)
        lid_memory.append(current_visible)
        
        # Calculate Mode (Most stable count in buffer)
        stable_count = max(set(lid_memory), key=lid_memory.count)

        # 1. AUTO-CALIBRATION
        if not calibrated and len(lid_memory) == BUFFER_SIZE:
            total_inventory = stable_count
            calibrated = True
            print(f"system ready!: {total_inventory} units")

        # 2. STACK ARRIVAL (+5)
        # Only triggers if 0 lids were present for a while, then stable lids appear
        if calibrated and stable_count > 0 and last_stable_count == 0:
            total_inventory += STACK_SIZE
            print("stack added")

        # 3. PRECISION PICK (-1)
        # Logic: If hand overlaps any lid, system is "ARMED"
        hand_contact = any(is_overlapping(h, l) for h in hands for l in lids)
        
        if hand_contact:
            hand_touching_active = True
        
        # If armed and count drops, increment missing counter
        if hand_touching_active and current_visible < stable_count:
            missing_count += 1
        else:
            # If hand moves away and count didn't drop, disarm
            if not hand_contact:
                hand_touching_active = False
                missing_count = 0

        # Confirm Pick after PICK_CONFIDENCE frames (Prevents scuffed flickers)
        if missing_count >= PICK_CONFIDENCE:
            total_inventory -= 1
            missing_count = 0
            hand_touching_active = False
            print(f"unit REMOVED! total count: {total_inventory}")

        last_stable_count = stable_count

        # --- PRO UI OVERLAY ---
        # Dark header bar
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (20, 20, 20), -1)
        # Status Light
        status_color = (0, 255, 0) if calibrated else (0, 165, 255)
        cv2.circle(frame, (25, 35), 8, status_color, -1)
        
        cv2.putText(frame, f"All-CLAD Lid Count: {max(0, total_inventory)}", (50, 42), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 255), 1)
        cv2.putText(frame, f"Sensors: {stable_count} active", (frame.shape[1]-200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, f"latency: {inference_speed:.1f}ms", (frame.shape[1]-200, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        display_frame = frame.copy()

    # Pause Logic
    if is_paused:
        cv2.putText(display_frame, "paused", (frame.shape[1]//4, frame.shape[2]//2), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 0, 255), 2)

    cv2.imshow("lid line declaration", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    if key == ord('p'): is_paused = not is_paused
    if key == ord('r'): # Full hardware reset
        total_inventory = 0
        calibrated = False
        lid_memory.clear()

vs.stop()
cv2.destroyAllWindows()
