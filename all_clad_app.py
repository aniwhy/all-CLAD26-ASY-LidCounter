import streamlit as st
import cv2
from ultralytics import YOLO
import time
import pandas as pd
from collections import deque

# --- ALL-CLAD BRANDED UI ---
st.set_page_config(page_title="All-Clad AI Inventory", layout="wide")

# Professional dark-industrial styling
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_name_with_html=True)

st.title("🏭 All-Clad Inventory Systems")
st.write("Production Line 1 | Lid Detection Unit V4.0")

# --- SIDEBAR CONTROLS ---
st.sidebar.image("https://www.all-clad.com/static/version1675344383/frontend/AllClad/default/en_US/images/logo.svg", width=150)
st.sidebar.header("System Control")
conf_threshold = st.sidebar.slider("AI Confidence", 0.3, 0.9, 0.5)
reset_btn = st.sidebar.button("Hard Reset Inventory")

# Use session state to keep data alive during reruns
if 'total_inv' not in st.session_state or reset_btn:
    st.session_state.total_inv = 0
if 'log_data' not in st.session_state:
    st.session_state.log_data = []

# --- CORE AI MODEL ---
@st.cache_resource
def get_model():
    return YOLO('lidDetection.pt')

model = get_model()

# --- LAYOUT ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Sensor Feed")
    frame_window = st.empty()  # Placeholder for the video

with col2:
    st.subheader("Real-Time Metrics")
    metric_box = st.empty()    # Placeholder for counters
    st.subheader("Event History")
    history_box = st.empty()   # Placeholder for logs

# --- VIDEO INITIALIZATION ---
PHONE_IP_URL = "http://192.168.0.52:4747/video"

# We open the capture once and store it in session state
if 'cap' not in st.session_state:
    cap = cv2.VideoCapture(PHONE_IP_URL)
    if not cap.isOpened():
        # Fallback to demo video if phone isn't connected
        cap = cv2.VideoCapture("test_video.mp4")
    st.session_state.cap = cap

cap = st.session_state.cap

# --- LOGIC CONSTANTS ---
BUFFER_SIZE = 15
lid_memory = deque(maxlen=BUFFER_SIZE)
last_stable_count = 0
hand_touching_active = False
calibrated = False
missing_count = 0
PICK_CONFIDENCE = 10

# --- MAIN LOOP ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        # If video ends, restart it (great for the demo video loop)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    results = model(frame, conf=conf_threshold, imgsz=640, verbose=False)
    
    hands = []
    lids = []
    
    for r in results:
        for box in r.boxes:
            coords = box.xyxy[0].tolist()
            label = model.names[int(box.cls[0])]
            if label == 'hand': hands.append(coords)
            else: lids.append(coords)
            
            # Annotate frame
            x1, y1, x2, y2 = map(int, coords)
            color = (255, 191, 0) if label == 'hand' else (0, 255, 127)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Logic Calculations
    current_visible = len(lids)
    lid_memory.append(current_visible)
    stable_count = max(set(lid_memory), key=lid_memory.count) if lid_memory else 0

    if not calibrated and len(lid_memory) == BUFFER_SIZE:
        st.session_state.total_inv = stable_count
        calibrated = True

    # Detection: Stack Added
    if calibrated and stable_count > last_stable_count and last_stable_count == 0:
        st.session_state.total_inv += 5
        st.session_state.log_data.insert(0, f"{time.strftime('%H:%M:%S')} - Stack Added (+5)")

    # Detection: Hand Pick
    hand_contact = any(
        not (h[2] < l[0] or h[0] > l[2] or h[3] < l[1] or h[1] > l[3]) 
        for h in hands for l in lids
    )
    
    if hand_contact: 
        hand_touching_active = True
    
    if hand_touching_active and current_visible < stable_count:
        missing_count += 1
    
    if missing_count >= PICK_CONFIDENCE:
        st.session_state.total_inv -= 1
        st.session_state.log_data.insert(0, f"{time.strftime('%H:%M:%S')} - Unit Removed (-1)")
        missing_count = 0
        hand_touching_active = False

    last_stable_count = stable_count

    # --- UI UPDATES ---
    with metric_box.container():
        st.metric("CURRENT INVENTORY", f"{st.session_state.total_inv} Units")
        status = "CONNECTED" if hand_contact else "IDLE"
        st.write(f"Sensors: {stable_count} Active | Hand Link: {status}")

    frame_window.image(frame, channels="BGR", use_column_width=True)
    history_box.text("\n".join(st.session_state.log_data[:8]))

    # Small sleep to prevent CPU spiking
    time.sleep(0.01)
