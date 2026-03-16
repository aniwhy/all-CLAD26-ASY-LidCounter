import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
from ultralytics import YOLO
import time
import av

# --- ALL-CLAD BRANDED UI ---
st.set_page_config(page_title="All-Clad Lid Inventory", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

st.title("All-Clad Lid Inventory Tracking")
st.write("Lid Count Tracking | Anirudh Yuvaraj")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("System Control")
st.sidebar.image("logo.png", width=150)
conf_threshold = st.sidebar.slider("AI Confidence", 0.3, 0.9, 0.5)
reset_btn = st.sidebar.button("Hard Reset Inventory Count")

# --- SESSION STATE ---
if 'total_inv' not in st.session_state or reset_btn:
    st.session_state.total_inv = 0
if 'log_data' not in st.session_state or reset_btn:
    st.session_state.log_data = []
if 'lid_memory' not in st.session_state or reset_btn:
    st.session_state.lid_memory = []
if 'last_stable_count' not in st.session_state or reset_btn:
    st.session_state.last_stable_count = 0
if 'hand_touching_active' not in st.session_state or reset_btn:
    st.session_state.hand_touching_active = False
if 'calibrated' not in st.session_state or reset_btn:
    st.session_state.calibrated = False
if 'missing_count' not in st.session_state or reset_btn:
    st.session_state.missing_count = 0

# --- MODEL ---
@st.cache_resource
def get_model():
    return YOLO('lidDetection.pt')

model = get_model()

# --- CONSTANTS ---
BUFFER_SIZE = 15
PICK_CONFIDENCE = 10

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

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        results = model(img, conf=self.conf, imgsz=640, verbose=False)

        hands = []
        lids = []

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

        # --- LOGIC ---
        current_visible = len(lids)
        st.session_state.lid_memory.append(current_visible)
        if len(st.session_state.lid_memory) > BUFFER_SIZE:
            st.session_state.lid_memory.pop(0)

        stable_count = (
            max(set(st.session_state.lid_memory), key=st.session_state.lid_memory.count)
            if st.session_state.lid_memory else 0
        )

        if not st.session_state.calibrated and len(st.session_state.lid_memory) == BUFFER_SIZE:
            st.session_state.total_inv = stable_count
            st.session_state.calibrated = True

        if (st.session_state.calibrated
                and stable_count > st.session_state.last_stable_count
                and st.session_state.last_stable_count == 0):
            st.session_state.total_inv += 5
            st.session_state.log_data.insert(0, f"{time.strftime('%H:%M:%S')} - Stack Added (+5)")

        hand_contact = any(
            not (h[2] < l[0] or h[0] > l[2] or h[3] < l[1] or h[1] > l[3])
            for h in hands for l in lids
        )

        if hand_contact:
            st.session_state.hand_touching_active = True

        if st.session_state.hand_touching_active and current_visible < stable_count:
            st.session_state.missing_count += 1

        if st.session_state.missing_count >= PICK_CONFIDENCE:
            st.session_state.total_inv -= 1
            st.session_state.log_data.insert(0, f"{time.strftime('%H:%M:%S')} - Unit Removed (-1)")
            st.session_state.missing_count = 0
            st.session_state.hand_touching_active = False

        st.session_state.last_stable_count = stable_count

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# --- LAYOUT ---
col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("Real-Time Metrics")
    st.metric("Current Inventory", f"{st.session_state.total_inv} Units")
    st.subheader("Event History")
    st.text("\n".join(st.session_state.log_data[:8]))

with col1:
    st.subheader("Live Camera Feed")

    if mode == "📱 Live Camera (WebRTC)":
        ctx = webrtc_streamer(
            key="lid-detector",
            video_processor_factory=LidDetector,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={"video": True, "audio": False},
        )
        if ctx.video_processor:
            ctx.video_processor.conf = conf_threshold
        status = "ACTIVE" if (ctx and ctx.state.playing) else "IDLE"
        st.write(f"Camera: {status} | Calibrated: {st.session_state.calibrated}")

    else:
        # Demo video mode using OpenCV loop
        frame_window = st.empty()
        if 'demo_cap' not in st.session_state:
            st.session_state.demo_cap = cv2.VideoCapture("test_video.mp4")
        cap = st.session_state.demo_cap

        stop_btn = st.button("⏹ Stop Demo")
        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

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

            current_visible = len(lids)
            st.session_state.lid_memory.append(current_visible)
            if len(st.session_state.lid_memory) > BUFFER_SIZE:
                st.session_state.lid_memory.pop(0)

            stable_count = (
                max(set(st.session_state.lid_memory), key=st.session_state.lid_memory.count)
                if st.session_state.lid_memory else 0
            )

            if not st.session_state.calibrated and len(st.session_state.lid_memory) == BUFFER_SIZE:
                st.session_state.total_inv = stable_count
                st.session_state.calibrated = True

            if (st.session_state.calibrated
                    and stable_count > st.session_state.last_stable_count
                    and st.session_state.last_stable_count == 0):
                st.session_state.total_inv += 5
                st.session_state.log_data.insert(0, f"{time.strftime('%H:%M:%S')} - Stack Added (+5)")

            hand_contact = any(
                not (h[2] < l[0] or h[0] > l[2] or h[3] < l[1] or h[1] > l[3])
                for h in hands for l in lids
            )
            if hand_contact:
                st.session_state.hand_touching_active = True
            if st.session_state.hand_touching_active and current_visible < stable_count:
                st.session_state.missing_count += 1
            if st.session_state.missing_count >= PICK_CONFIDENCE:
                st.session_state.total_inv -= 1
                st.session_state.log_data.insert(0, f"{time.strftime('%H:%M:%S')} - Unit Removed (-1)")
                st.session_state.missing_count = 0
                st.session_state.hand_touching_active = False

            st.session_state.last_stable_count = stable_count
            frame_window.image(frame, channels="BGR", width='stretch')
            time.sleep(0.01)
