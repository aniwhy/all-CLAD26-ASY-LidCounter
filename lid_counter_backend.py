import cv2
from ultralytics import YOLO
import threading
import time

# Anirudh Yuvaraj, Jonathan Philip
# All-CLAD Project, 2026
# Production-Grade Inventory System (V8.0)

model = YOLO('lidDetection.pt')
print(f"[MODEL] Classes: {model.names}")

# --- Configuration ---
HOLD_CONFIRM = 8     # frames hand+lid_handle must overlap to confirm grab
GONE_CONFIRM = 10    # frames hand must be gone before committing event
CONF_THRESH  = 0.55  # high threshold to suppress printed-diagram false positives

# --- Persistent inventory ---
total_inventory = 0

# --- State machine ---
STATE_WATCHING = "WATCHING"
STATE_HAND_IN  = "HAND_IN"
STATE_HOLDING  = "HOLDING"
STATE_LEAVING  = "LEAVING"

state        = STATE_WATCHING
hold_frames  = 0
gone_frames  = 0
hand_had_lid = False
holding_lid  = False
exit_has_lid = False

is_paused     = False
display_frame = None

# Brief flash message for manual key actions
flash_msg     = ""
flash_until   = 0.0


def boxes_overlap(b1, b2):
    return not (b1[2] < b2[0] or b1[0] > b2[2] or
                b1[3] < b2[1] or b1[1] > b2[3])


class VideoStream:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self._update, daemon=True).start()
        return self

    def _update(self):
        while not self.stopped:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()


# School wifi: "http://192.168.137.18:4747/video"
vs = VideoStream("http://192.168.0.52:4747/video").start()
time.sleep(2.0)

cv2.namedWindow("All-CLAD Lid Detection")

while True:
    if not is_paused:
        ret, frame = vs.read()
        if not ret or frame is None:
            break

        fh, fw = frame.shape[:2]

        t0 = time.time()
        results = model(frame, conf=CONF_THRESH, imgsz=640, verbose=False)
        latency = (time.time() - t0) * 1000

        hands, lids = [], []
        for r in results:
            for box in r.boxes:
                coords = box.xyxy[0].tolist()
                label  = model.names[int(box.cls[0])]
                conf   = float(box.conf[0])

                if label == 'hand':
                    hands.append(coords)
                elif label == 'lid_handle':
                    lids.append(coords)

                x1, y1, x2, y2 = map(int, coords)
                color = (255, 191, 0) if label == 'hand' else (0, 255, 127)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        hand_present   = len(hands) > 0
        lid_present    = len(lids)  > 0
        visible_count  = len(lids)
        hand_holds_lid = hand_present and lid_present and any(
            boxes_overlap(h, l) for h in hands for l in lids
        )

        # ── STATE MACHINE ──────────────────────────────────────────────────

        if state == STATE_WATCHING:
            if hand_present:
                hand_had_lid = lid_present
                hold_frames  = 1
                state        = STATE_HAND_IN
                print(f"[HAND IN]  lid already visible: {hand_had_lid}")

        elif state == STATE_HAND_IN:
            if not hand_present:
                state = STATE_WATCHING
                print("[IGNORE]  hand left too fast")
            else:
                if hand_holds_lid:
                    hold_frames += 1
                if hold_frames >= HOLD_CONFIRM:
                    holding_lid = hand_holds_lid
                    state       = STATE_HOLDING
                    print(f"[HOLDING]  lid confirmed in hand: {holding_lid}")

        elif state == STATE_HOLDING:
            if not hand_present:
                exit_has_lid = lid_present
                gone_frames  = 1
                state        = STATE_LEAVING
                print(f"[LEAVING]  lid still visible: {exit_has_lid}")
            else:
                if hand_holds_lid:
                    holding_lid = True

        elif state == STATE_LEAVING:
            if hand_present:
                hold_frames = HOLD_CONFIRM
                holding_lid = hand_holds_lid
                state       = STATE_HOLDING
                print("[BACK]  hand returned")
            else:
                gone_frames  += 1
                exit_has_lid  = lid_present

                if gone_frames >= GONE_CONFIRM:
                    if not hand_had_lid and exit_has_lid:
                        total_inventory += 1
                        print(f"[+1] lid placed in. Inventory: {total_inventory}")
                    elif holding_lid and not exit_has_lid:
                        total_inventory = max(0, total_inventory - 1)
                        print(f"[-1] lid taken out. Inventory: {total_inventory}")
                    else:
                        print(f"[NO CHANGE] had={hand_had_lid} "
                              f"held={holding_lid} exit={exit_has_lid}")

                    state        = STATE_WATCHING
                    hold_frames  = 0
                    gone_frames  = 0
                    hand_had_lid = False
                    holding_lid  = False
                    exit_has_lid = False

        # ── UI HEADER ──────────────────────────────────────────────────────
        cv2.rectangle(frame, (0, 0), (fw, 70), (20, 20, 20), -1)

        border_color = {
            STATE_WATCHING: None,
            STATE_HAND_IN:  (0, 220, 255),
            STATE_HOLDING:  (0, 140, 255),
            STATE_LEAVING:  (0, 255, 80),
        }.get(state)
        if border_color:
            cv2.rectangle(frame, (0, 0), (fw, fh), border_color, 3)

        cv2.circle(frame, (25, 35), 8, (0, 255, 0), -1)
        cv2.putText(frame,
                    f"All-CLAD Inventory: {total_inventory}",
                    (50, 42), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 255), 1)
        cv2.putText(frame,
                    f"State: {state}   H:{hold_frames}/{HOLD_CONFIRM}  G:{gone_frames}/{GONE_CONFIRM}  Visible:{visible_count}",
                    (10, fh - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        cv2.putText(frame,
                    f"Latency: {latency:.1f}ms",
                    (fw - 180, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # Flash message (manual key actions)
        if time.time() < flash_until:
            cv2.putText(frame, flash_msg,
                        (fw // 2 - 160, fh // 2),
                        cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 255), 2)

        display_frame = frame.copy()

    if is_paused and display_frame is not None:
        dh, dw = display_frame.shape[:2]
        cv2.putText(display_frame, "PAUSED", (dw // 4, dh // 2),
                    cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 0, 255), 2)

    if display_frame is not None:
        cv2.imshow("All-CLAD Lid Detection", display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('p'):
        is_paused = not is_paused

    elif key == ord('=') or key == ord('+'):
        # Manual +1
        total_inventory += 1
        flash_msg   = f"MANUAL +1  ->  {total_inventory}"
        flash_until = time.time() + 1.5
        print(f"[MANUAL +1] Inventory: {total_inventory}")

    elif key == ord('-'):
        # Manual -1
        total_inventory = max(0, total_inventory - 1)
        flash_msg   = f"MANUAL -1  ->  {total_inventory}"
        flash_until = time.time() + 1.5
        print(f"[MANUAL -1] Inventory: {total_inventory}")

    elif key == ord('s'):
        # Sync: ADD currently visible lid_handle count to inventory
        if display_frame is not None:
            total_inventory += visible_count
            flash_msg   = f"ADDED {visible_count}  ->  {total_inventory}"
            flash_until = time.time() + 1.5
            print(f"[SYNC] Added {visible_count} visible lids. Inventory: {total_inventory}")

    elif key == ord('r'):
        total_inventory = 0
        state           = STATE_WATCHING
        hold_frames     = 0
        gone_frames     = 0
        hand_had_lid    = False
        holding_lid     = False
        exit_has_lid    = False
        flash_msg       = "RESET"
        flash_until     = time.time() + 1.5
        print("[RESET]")

vs.stop()
cv2.destroyAllWindows()
