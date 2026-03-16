import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
from ultralytics import YOLO
import time
import av
import threading

st.set_page_config(page_title="All-Clad Lid Inventory", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

st.title("All-Clad Lid Inventory Tracking")
st.write("Lid Count Tracking | Anirudh Yuvaraj")

# --- SIDEBAR ---
st.sidebar.header("System Control")
st.sidebar.image("logo.png", width=150)
conf_threshold = st.sidebar.slider("AI Confidence", 0.3, 0.9, 0.5)
reset_btn = st.sidebar.button("Hard Reset Inventory Count")
mode = st.sidebar.radio("Input Mode", ["Live Camera (WebRTC)", "Demo Video"])

# --- MODEL ---
@st.cache_resource
def get_model():
    return YOLO('lidDetection.pt')

model = get_model()

BUFFER_SIZE = 15

RTC_CONFIG = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
})


def run_logic(current_visible, hand_contact, s):
    if not hand_contact:
        s["lid_memory"].append(current_visible)
        if len(s["lid_memory"]) > BUFFER_SIZE:
            s["lid_memory"].pop(0)

    if not s["lid_memory"]:
        return

    stable_count = max(set(s["lid_memory"]), key=s["lid_memory"].count)

    if not s["calibrated"] and len(s["lid_memory"]) == BUFFER_SIZE:
        s["total_inv"] = stable_count
        s["last_stable_count"] = stable_count
        s["calibrated"] = True
        s["log"].insert(0, f"{time.strftime('%H:%M:%S')} - Calibrated ({stable_count} lids)")
        return

    if not s["calibrated"]:
        return

    if (not hand_contact
            and stable_count > s["last_stable_count"]
            and s["last_stable_count"] == 0):
        s["total_inv"] += stable_count
        s["log"].insert(0, f"{time.strftime('%H:%M:%S')} - Stack Added (+{stable_count})")
        s["last_stable_count"] = stable_count

    if hand_contact and not s["prev_hand"]:
        s["touching"] = True
        s["grab_count"] = s["last_stable_count"]
        s["confirm"] = 0

    if not hand_contact and s["prev_hand"] and s["touching"]:
        s["confirm"] = 0

    if not hand_contact and s["touching"]:
        if current_visible < s["grab_count"]:
            s["confirm"] += 1
        else:
            s["confirm"] = 0
            s["touching"] = False

        if s["confirm"] >= 8:
            removed = s["grab_count"] - current_visible
            if removed > 0:
                s["total_inv"] -= removed
                s["log"].insert(0, f"{time.strftime('%H:%M:%S')} - Removed (-{removed})")
                s["last_stable_count"] = current_visible
                s["log"] = s["log"][:20]
            s["touching"] = False
            s["confirm"] = 0

    s["prev_hand"] = hand_contact

    if not hand_contact and not s["touching"]:
        s["last_stable_count"] = stable_count


def make_state():
    return {
        "lid_memory": [],
        "last_stable_count": -1,
        "touching": False,
        "prev_hand": False,
        "grab_count": 0,
        "confirm": 0,
        "calibrated": False,
        "total_inv": 0,
        "log": [],
    }


# --- VIDEO PROCESSOR ---
class LidDetector(VideoProcessorBase):
    def __init__(self):
        self.conf = 0.5
        self.lock = threading.Lock()
        self.s = make_state()

    def reset(self):
        with self.lock:
            self.s = make_state()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img, conf=self.conf, imgsz=640, verbose=False)

        hands, lids = [], []
        for r in results:
            for box in r.boxes:
                coords = box.xyxy[0].tolist()
                label = model.names[int(box.cls[0])]
                if label == 'hand':
                    hands.append(coords)
                else:
                    lids.append(coords)
                x1, y1, x2, y2 = map(int, coords)
                color = (255, 191, 0) if label == 'hand' else (0, 255, 127)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        current_visible = len(lids)
        hand_contact = any(
            not (h[2] < l[0] or h[0] > l[2] or h[3] < l[1] or h[1] > l[3])
            for h in hands for l in lids
        )

        with self.lock:
            run_logic(current_visible, hand_contact, self.s)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def get_display(self):
        with self.lock:
            return self.s["total_inv"], list(self.s["log"]), self.s["calibrated"]


# --- LAYOUT ---
col1, col2 = st.columns([2, 1])

# Placeholders so we can update metrics without rerunning the whole page
with col2:
    st.subheader("Real-Time Metrics")
    metric_placeholder = st.empty()
    cal_placeholder = st.empty()
    st.subheader("Event History")
    log_placeholder = st.empty()

with col1:
    st.subheader("Live Camera Feed")

    if mode == "Live Camera (WebRTC)":
        if 'demo_cap' in st.session_state:
            st.session_state.demo_cap.release()
            del st.session_state.demo_cap

        ctx = webrtc_streamer(
            key="lid-detector",
            video_processor_factory=LidDetector,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={"video": True, "audio": False},
        )

        if ctx.video_processor:
            ctx.video_processor.conf = conf_threshold
            if reset_btn:
                ctx.video_processor.reset()

            # Poll loop — updates placeholders without full page rerun
            while True:
                total_inv, log, calibrated = ctx.video_processor.get_display()
                metric_placeholder.metric("Current Inventory", f"{total_inv} Units")
                cal_placeholder.write(f"Calibrated: {calibrated}")
                log_placeholder.text("\n".join(log[:8]))
                time.sleep(0.3)
        else:
            metric_placeholder.metric("Current Inventory", "-- Units")
            cal_placeholder.write("Waiting for camera...")

    else:
        # Demo mode
        s = make_state()
        if reset_btn:
            s = make_state()

        frame_window = st.empty()
        status_placeholder = st.empty()

        if 'demo_cap' not in st.session_state or reset_btn:
            if 'demo_cap' in st.session_state:
                st.session_state.demo_cap.release()
            st.session_state.demo_cap = cv2.VideoCapture("demo_video.mp4")

        col_start, col_stop = st.columns(2)
        with col_start:
            start = st.button("▶ Start Demo")
        with col_stop:
            stop = st.button("⏹ Stop Demo")

        if start:
            st.session_state.demo_running = True
        if stop:
            st.session_state.demo_running = False

        if st.session_state.get('demo_running', False):
            cap = st.session_state.demo_cap
            while True:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()

                if ret:
                    results = model(frame, conf=conf_threshold, imgsz=640, verbose=False)
                    hands, lids = [], []
                    for r in results:
                        for box in r.boxes:
                            coords = box.xyxy[0].tolist()
                            label = model.names[int(box.cls[0])]
                            if label == 'hand':
                                hands.append(coords)
                            else:
                                lids.append(coords)
                            x1, y1, x2, y2 = map(int, coords)
                            color = (255, 191, 0) if label == 'hand' else (0, 255, 127)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame, label, (x1, y1 - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                    hand_contact = any(
                        not (h[2] < l[0] or h[0] > l[2] or h[3] < l[1] or h[1] > l[3])
                        for h in hands for l in lids
                    )
                    run_logic(len(lids), hand_contact, s)

                    frame_window.image(frame, channels="BGR", width='stretch')
                    metric_placeholder.metric("Current Inventory", f"{s['total_inv']} Units")
                    cal_placeholder.write(f"Calibrated: {s['calibrated']}")
                    log_placeholder.text("\n".join(s['log'][:8]))
                    time.sleep(0.05)
        else:
            metric_placeholder.metric("Current Inventory", "-- Units")
            cal_placeholder.write("Press ▶ Start Demo to begin.")
