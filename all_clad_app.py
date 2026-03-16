import streamlit as st
import subprocess
import sys
subprocess.run([sys.executable, "-m", "pip", "uninstall", "opencv-python", "-y"], 
               capture_output=True)
import cv2
from ultralytics import YOLO
import time
import pandas as pd

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
st.sidebar.image("https://arnoldstreet.com/project/all-clad-cookware", width=150)
st.sidebar.header("System Control")
conf_threshold = st.sidebar.slider("AI Confidence", 0.3, 0.9, 0.5)
reset_btn = st.sidebar.button("Hard Reset Inventory Count")

# --- SESSION STATE INITIALIZATION ---
# Using plain lists/primitives only — deque is not safely serializable in session_state
if 'total_inv' not in st.session_state or reset_btn:
    st.session_state.total_inv = 0
if 'log_data' not in st.session_state or reset_btn:
    st.session_state.log_data = []
if 'lid_memory' not in st.session_state or reset_btn:
    st.session_state.lid_memory = []          # plain list, capped manually at 15
if 'last_stable_count' not in st.session_state or reset_btn:
    st.session_state.last_stable_count = 0
if 'hand_touching_active' not in st.session_state or reset_btn:
    st.session_state.hand_touching_active = False
if 'calibrated' not in st.session_state or reset_btn:
    st.session_state.calibrated = False
if 'missing_count' not in st.session_state or reset_btn:
    st.session_state.missing_count = 0

# --- CORE AI MODEL ---
@st.cache_resource
def get_model():
    return YOLO('lidDetection.pt')

model = get_model()

# --- LAYOUT ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Sensor Feed")
    frame_window = st.empty()

with col2:
    st.subheader("Real-Time Metrics")
    metric_box = st.empty()
    st.subheader("Event History")
    history_box = st.empty()

# --- VIDEO INITIALIZATION ---
PHONE_IP_URL = "http://192.168.0.52:4747/video"
BUFFER_SIZE = 15
PICK_CONFIDENCE = 10

if 'cap' not in st.session_state:
    cap = cv2.VideoCapture(PHONE_IP_URL)
    if not cap.isOpened():
        cap = cv2.VideoCapture("test_video.mp4")
    st.session_state.cap = cap

cap = st.session_state.cap

# --- MAIN LOOP ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    results = model(frame, conf=conf_threshold, imgsz=640, verbose=False)

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
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # --- LOGIC ---
    current_visible = len(lids)

    # Append and manually cap at BUFFER_SIZE (replaces deque)
    st.session_state.lid_memory.append(current_visible)
    if len(st.session_state.lid_memory) > BUFFER_SIZE:
        st.session_state.lid_memory.pop(0)

    stable_count = (
        max(set(st.session_state.lid_memory), key=st.session_state.lid_memory.count)
        if st.session_state.lid_memory else 0
    )

    # Calibrate on first full buffer
    if not st.session_state.calibrated and len(st.session_state.lid_memory) == BUFFER_SIZE:
        st.session_state.total_inv = stable_count
        st.session_state.calibrated = True

    # Detection: Stack Added
    if (st.session_state.calibrated
            and stable_count > st.session_state.last_stable_count
            and st.session_state.last_stable_count == 0):
        st.session_state.total_inv += 5
        st.session_state.log_data.insert(0, f"{time.strftime('%H:%M:%S')} - Stack Added (+5)")

    # Detection: Hand Pick
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

    # --- UI UPDATES ---
    with metric_box.container():
        st.metric("CURRENT INVENTORY", f"{st.session_state.total_inv} Units")
        status = "CONNECTED" if hand_contact else "IDLE"
        st.write(f"Sensors: {stable_count} Active | Hand Link: {status}")

    frame_window.image(frame, channels="BGR", width='stretch')  # fixed deprecated param
    history_box.text("\n".join(st.session_state.log_data[:8]))

    time.sleep(0.01)
