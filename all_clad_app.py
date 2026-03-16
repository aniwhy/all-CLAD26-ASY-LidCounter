import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
from ultralytics import YOLO
import time
import av

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

# --- SESSION STATE ---
defaults = {
    'total_inv': 0,
    'log_data': [],
    'lid_memory': [],
    'last_stable_count': -1,
    'hand_touching_active': False,
    'calibrated': False,
    'demo_running': False,
    'prev_hand_contact': False,
    'count_before_grab': 0,
    'post_grab_confirm': 0,
}
for k, v in defaults.items():
    if k not in st.session_state or reset_btn:
        st.session_state[k] = v

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

# --- VIDEO PROCESSOR ---
class LidDetector(VideoProcessorBase):
    def __init__(self):
        self.conf = 0.5
        self.result = {"current_visible": 0, "hand_contact": False}

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

        hand_contact = any(
            not (h[2] < l[0] or h[0] > l[2] or h[3] < l[1] or h[1] > l[3])
            for h in hands for l in lids
        )
        self.result = {"current_visible": len(lids), "hand_contact": hand_contact}
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def run_logic(current_visible, hand_contact):
    ss = st.session_state

    # Only update lid memory when hand is not in frame
    # so camera wobble during a grab doesn't corrupt the buffer
    if not hand_contact:
        ss.lid_memory.append(current_visible)
        if len(ss.lid_memory) > BUFFER_SIZE:
            ss.lid_memory.pop(0)

    if not ss.lid_memory:
        return

    stable_count = max(set(ss.lid_memory), key=ss.lid_memory.count)

    # Calibrate once buffer is full
    if not ss.calibrated and len(ss.lid_memory) == BUFFER_SIZE:
        ss.total_inv = stable_count
        ss.last_stable_count = stable_count
        ss.calibrated = True
        ss.log_data.insert(0, f"{time.strftime('%H:%M:%S')} - Calibrated ({stable_count} lids)")
        return

    if not ss.calibrated:
        return

    # Stack added: only trigger when hand free and count rises from 0
    if not hand_contact and stable_count > ss.last_stable_count and ss.last_stable_count == 0:
        ss.total_inv += stable_count
        ss.log_data.insert(0, f"{time.strftime('%H:%M:%S')} - Stack Added (+{stable_count})")
        ss.last_stable_count = stable_count

    # Hand just touched — snapshot the stable count before anything moves
    if hand_contact and not ss.prev_hand_contact:
        ss.hand_touching_active = True
        ss.count_before_grab = ss.last_stable_count
        ss.post_grab_confirm = 0

    # Hand just left — start confirmation window
    if not hand_contact and ss.prev_hand_contact and ss.hand_touching_active:
        ss.post_grab_confirm = 0

    # Confirmation window: count consecutive frames where lid count stays lower
    if not hand_contact and ss.hand_touching_active:
        if current_visible < ss.count_before_grab:
            ss.post_grab_confirm += 1
        else:
            # Count recovered — false alarm (camera moved, lid put back, etc.)
            ss.post_grab_confirm = 0
            ss.hand_touching_active = False

        # Only subtract after 5 consistent frames confirming lid is gone
        if ss.post_grab_confirm >= 5:
            removed = ss.count_before_grab - current_visible
            if removed > 0:
                ss.total_inv -= removed
                ss.log_data.insert(0, f"{time.strftime('%H:%M:%S')} - Unit Removed (-{removed})")
                ss.last_stable_count = current_visible
            ss.hand_touching_active = False
            ss.post_grab_confirm = 0

    ss.prev_hand_contact = hand_contact

    # Update baseline only when hand is completely clear
    if not hand_contact and not ss.hand_touching_active:
        ss.last_stable_count = stable_count


# --- LAYOUT ---
col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("Real-Time Metrics")
    st.metric("Current Inventory", f"{st.session_state.total_inv} Units")
    st.write(f"Calibrated: {st.session_state.calibrated}")
    st.subheader("Event History")
    st.text("\n".join(st.session_state.log_data[:8]))

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
            r = ctx.video_processor.result
            run_logic(r["current_visible"], r["hand_contact"])
            status = "ACTIVE" if ctx.state.playing else "IDLE"
            st.write(f"Camera: {status}")
            time.sleep(0.2)
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
                run_logic(len(lids), hand_contact)
                frame_window.image(frame, channels="BGR", width='stretch')
                time.sleep(0.05)
                st.rerun()
        else:
            st.info("Press ▶ Start Demo to begin.")
