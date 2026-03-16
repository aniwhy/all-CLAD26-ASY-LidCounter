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

# --- CACHED DETECTOR STATE ---
# This survives reruns unlike session_state for the WebRTC processor
@st.cache_resource
def get_detector_state():
    return {
        "lock": threading.Lock(),
        "lid_memory": [],
        "last_stable_count": -1,
        "hand_touching_active": False,
        "prev_hand_contact": False,
        "count_before_grab": 0,
        "post_grab_confirm": 0,
        "calibrated": False,
        "total_inv": 0,
        "log_data": [],
    }

detector_state = get_detector_state()

# Reset cached state if button pressed
if reset_btn:
    with detector_state["lock"]:
        detector_state.update({
            "lid_memory": [],
            "last_stable_count": -1,
            "hand_touching_active": False,
            "prev_hand_contact": False,
            "count_before_grab": 0,
            "post_grab_confirm": 0,
            "calibrated": False,
            "total_inv": 0,
            "log_data": [],
        })

# --- SESSION STATE for demo mode ---
demo_defaults = {
    'total_inv': 0,
    'log_data': [],
    'calibrated': False,
    'demo_running': False,
    'lid_memory': [],
    'last_stable_count': -1,
    'hand_touching_active': False,
    'prev_hand_contact': False,
    'count_before_grab': 0,
    'post_grab_confirm': 0,
}
for k, v in demo_defaults.items():
    if k not in st.session_state or reset_btn:
        st.session_state[k] = v

BUFFER_SIZE = 15

RTC_CONFIG = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
})


def run_logic(current_visible, hand_contact, state):
    """Works with either detector_state dict or st.session_state."""

    if not hand_contact:
        state["lid_memory"].append(current_visible)
        if len(state["lid_memory"]) > BUFFER_SIZE:
            state["lid_memory"].pop(0)

    if not state["lid_memory"]:
        return

    stable_count = max(set(state["lid_memory"]), key=state["lid_memory"].count)

    if not state["calibrated"] and len(state["lid_memory"]) == BUFFER_SIZE:
        state["total_inv"] = stable_count
        state["last_stable_count"] = stable_count
        state["calibrated"] = True
        state["log_data"].insert(0, f"{time.strftime('%H:%M:%S')} - Calibrated ({stable_count} lids)")
        return

    if not state["calibrated"]:
        return

    # Stack added
    if (not hand_contact
            and stable_count > state["last_stable_count"]
            and state["last_stable_count"] == 0):
        state["total_inv"] += stable_count
        state["log_data"].insert(0, f"{time.strftime('%H:%M:%S')} - Stack Added (+{stable_count})")
        state["last_stable_count"] = stable_count

    # Hand just touched
    if hand_contact and not state["prev_hand_contact"]:
        state["hand_touching_active"] = True
        state["count_before_grab"] = state["last_stable_count"]
        state["post_grab_confirm"] = 0

    # Hand just left
    if not hand_contact and state["prev_hand_contact"] and state["hand_touching_active"]:
        state["post_grab_confirm"] = 0

    # Confirmation window
    if not hand_contact and state["hand_touching_active"]:
        if current_visible < state["count_before_grab"]:
            state["post_grab_confirm"] += 1
        else:
            state["post_grab_confirm"] = 0
            state["hand_touching_active"] = False

        if state["post_grab_confirm"] >= 8:
            removed = state["count_before_grab"] - current_visible
            if removed > 0:
                state["total_inv"] -= removed
                state["log_data"].insert(0, f"{time.strftime('%H:%M:%S')} - Removed (-{removed})")
                state["last_stable_count"] = current_visible
                state["log_data"] = state["log_data"][:20]
            state["hand_touching_active"] = False
            state["post_grab_confirm"] = 0

    state["prev_hand_contact"] = hand_contact

    if not hand_contact and not state["hand_touching_active"]:
        state["last_stable_count"] = stable_count


# --- VIDEO PROCESSOR ---
class LidDetector(VideoProcessorBase):
    def __init__(self):
        self.conf = 0.5

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

        with detector_state["lock"]:
            run_logic(current_visible, hand_contact, detector_state)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# --- LAYOUT ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Camera Feed")

    if mode == "Live Camera (WebRTC)":
        if 'demo_cap' in st.session_state:
            st.session_state.demo_cap.release()
            del st.session_state.demo_cap
        st.session_state.demo_running = False

        ctx = webrtc_streamer(
            key="lid-detector",
            video_processor_factory=LidDetector,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={"video": True, "audio": False},
        )

        if ctx.video_processor:
            ctx.video_processor.conf = conf_threshold

        status = "ACTIVE" if (ctx and ctx.state.playing) else "IDLE"
        st.write(f"Camera: {status} | Calibrated: {detector_state['calibrated']}")
        time.sleep(0.3)
        st.rerun()

    else:
        frame_window = st.empty()

        if 'demo_cap' not in st.session_state:
            st.session_state.demo_cap = cv2.VideoCapture("demo_video.mp4")
            st.session_state.demo_running = True

        col_start, col_stop = st.columns(2)
        with col_start:
            if st.button("▶ Start Demo"):
                st.session_state.demo_running = True
        with col_stop:
            if st.button("⏹ Stop Demo"):
                st.session_state.demo_running = False

        if st.session_state.demo_running:
            cap = st.session_state.demo_cap
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
                run_logic(len(lids), hand_contact, st.session_state)
                frame_window.image(frame, channels="BGR", width='stretch')
                time.sleep(0.1)
                st.rerun()
        else:
            st.info("Press ▶ Start Demo to begin.")

with col2:
    st.subheader("Real-Time Metrics")
    # Show from the right source depending on mode
    if mode == "Live Camera (WebRTC)":
        with detector_state["lock"]:
            display_inv = detector_state["total_inv"]
            display_log = list(detector_state["log_data"])
            display_cal = detector_state["calibrated"]
    else:
        display_inv = st.session_state.total_inv
        display_log = st.session_state.log_data
        display_cal = st.session_state.calibrated

    st.metric("Current Inventory", f"{display_inv} Units")
    st.write(f"Calibrated: {display_cal}")
    st.subheader("Event History")
    st.text("\n".join(display_log[:8]))
