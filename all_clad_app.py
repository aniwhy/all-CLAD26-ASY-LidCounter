import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
from ultralytics import YOLO
import time
import av
import threading
import tempfile
import os

st.set_page_config(page_title="All-Clad Lid Inventory", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    /* Apply Inter to everything including WebRTC buttons */
    *, *::before, *::after,
    button, input, select, textarea,
    [class*="css"], [data-testid],
    .stButton > button,
    video, .webrtc-streamer { 
        font-family: 'Inter', sans-serif !important; 
    }

    .main { background-color: #0a0e14; }
    [data-testid="stAppViewContainer"] { background-color: #0a0e14; }
    [data-testid="stSidebar"] { 
        background-color: #0d1117; 
        border-right: 1px solid #1e2229; 
    }
    h1, h2, h3, p, label, .stMarkdown, 
    [data-testid="stText"] { color: #e6edf3 !important; }

    /* Animations */
    @keyframes pulse-red {
        0%   { box-shadow: 0 0 0 0 rgba(196,18,48,0.5); }
        70%  { box-shadow: 0 0 0 8px rgba(196,18,48,0); }
        100% { box-shadow: 0 0 0 0 rgba(196,18,48,0); }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(6px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes shimmer-red {
        0%   { background-position: 0%; }
        100% { background-position: 200%; }
    }
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50%       { opacity: 0.4; }
    }

    /* Metric card */
    .metric-card {
        background: linear-gradient(135deg, #12161c 0%, #161b22 100%);
        border: 1px solid #1e2229;
        border-radius: 10px;
        padding: 20px 24px;
        margin-bottom: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        animation: fadeInUp 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, #6b0015, #c41230, #e8394f, #c41230, #6b0015);
        background-size: 200%;
        animation: shimmer-red 4s linear infinite;
    }
    .metric-label {
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #4b5563 !important;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 52px;
        font-weight: 700;
        color: #f0f6fc !important;
        line-height: 1;
        font-family: 'JetBrains Mono', monospace !important;
    }
    .metric-sub {
        font-size: 11px;
        color: #2d3340;
        margin-top: 6px;
        letter-spacing: 1px;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-top: 10px;
        margin-right: 4px;
    }
    .badge-active {
        background: rgba(196,18,48,0.12);
        color: #e8394f;
        border: 1px solid rgba(196,18,48,0.3);
        animation: pulse-red 2s infinite;
    }
    .badge-idle {
        background: rgba(100,110,120,0.1);
        color: #6b7280;
        border: 1px solid rgba(100,110,120,0.2);
    }
    .badge-calibrated {
        background: rgba(160,160,160,0.08);
        color: #9ca3af;
        border: 1px solid rgba(160,160,160,0.15);
    }
    .badge-waiting {
        background: rgba(196,18,48,0.08);
        color: #c41230;
        border: 1px solid rgba(196,18,48,0.2);
        animation: blink 1.5s ease infinite;
    }

    /* Log card */
    .log-card {
        background: #0d1117;
        border: 1px solid #1e2229;
        border-radius: 8px;
        padding: 14px 16px;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 11px;
        color: #6b7280;
        min-height: 200px;
        line-height: 2;
    }
    .log-entry-remove { color: #e8394f; }
    .log-entry-add    { color: #9ca3af; }
    .log-entry-cal    { color: #4b5563; }

    /* Section headers */
    .section-header {
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #c41230 !important;
        border-bottom: 1px solid #1e2229;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }

    /* All buttons including WebRTC */
    button, .stButton > button,
    [data-testid="baseButton-secondary"],
    [data-testid="baseButton-primary"] {
        font-family: 'Inter', sans-serif !important;
        background-color: #12161c !important;
        color: #9ca3af !important;
        border: 1px solid #1e2229 !important;
        border-radius: 6px !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        transition: all 0.2s !important;
    }
    button:hover, .stButton > button:hover,
    [data-testid="baseButton-secondary"]:hover,
    [data-testid="baseButton-primary"]:hover {
        border-color: #c41230 !important;
        color: #e8394f !important;
        background-color: #1a1020 !important;
    }

    /* Slider */
    .stSlider > div > div > div > div { 
        background-color: #c41230 !important; 
    }

    /* Radio */
    .stRadio label { 
        font-size: 12px !important; 
        color: #9ca3af !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #0d1117;
        border: 1px dashed #1e2229;
        border-radius: 8px;
    }

    /* Hide default metric widget */
    [data-testid="stMetric"] { display: none; }

    /* Footer */
    .footer {
        margin-top: 48px;
        padding: 20px 0 8px 0;
        border-top: 1px solid #1e2229;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 10px;
    }
    .footer-left { font-size: 12px; color: #4b5563; }
    .footer-left strong { color: #9ca3af; }
    .footer-right { 
        font-size: 10px; 
        color: #2d3340; 
        letter-spacing: 2px; 
        text-transform: uppercase; 
    }
    .footer a { color: #c41230 !important; text-decoration: none; }
    .footer a:hover { color: #e8394f !important; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: #0d1117; }
    ::-webkit-scrollbar-thumb { background: #1e2229; border-radius: 2px; }
    </style>
""", unsafe_allow_html=True)


# ── Header ───────────────────────────────────────────────
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.image("logo.png", width=68)
with col_title:
    st.markdown("""
        <div style='padding-top:8px'>
            <div style='font-size:20px;font-weight:700;color:#f0f6fc;letter-spacing:1px'>
                All-Clad Lid Inventory
            </div>
            <div style='font-size:11px;color:#2d3340;letter-spacing:2px;
                        text-transform:uppercase;margin-top:3px'>
                Computer Vision Tracking &nbsp;·&nbsp; Production Line 1
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown(
    "<div style='height:1px;background:linear-gradient(90deg,transparent,#c41230,transparent);"
    "margin:14px 0 20px 0'></div>",
    unsafe_allow_html=True
)

# ── Sidebar ───────────────────────────────────────────────
st.sidebar.markdown("<div class='section-header'>System Control</div>", unsafe_allow_html=True)
conf_threshold = st.sidebar.slider("Detection Confidence", 0.3, 0.9, 0.5)
reset_btn = st.sidebar.button("⟳  Hard Reset")
mode = st.sidebar.radio("Input Mode", ["Live Camera (WebRTC)", "Demo Video", "Upload Video"])
st.sidebar.markdown("<div style='height:1px;background:#1e2229;margin:16px 0'></div>",
                    unsafe_allow_html=True)
st.sidebar.markdown("""
    <div style='font-size:10px;color:#2d3340;letter-spacing:1px;
                text-transform:uppercase;line-height:2.2'>
        Model · lidDetection.pt<br>
        Buffer · 15 frames<br>
        Confirm · 8 frames
    </div>
""", unsafe_allow_html=True)


# ── Model ─────────────────────────────────────────────────
@st.cache_resource
def get_model():
    return YOLO('lidDetection.pt')

model = get_model()

BUFFER_SIZE = 15
CONFIRM_THRESHOLD = 8

RTC_CONFIG = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
})


def make_state():
    return {
        # detection buffer — only filled when hand not in frame
        "lid_memory": [],
        # the last confirmed stable lid count (baseline)
        "baseline": 0,
        # whether we have calibrated at all yet
        "calibrated": False,
        # hand state
        "hand_was_present": False,
        "hand_is_present": False,
        # snapshot of baseline when hand first touched
        "count_at_touch": 0,
        # consecutive frames confirming lower count after hand left
        "confirm_frames": 0,
        # inventory
        "total_inv": 0,
        "log": [],
    }


def run_logic(current_visible, hand_contact, s):
    """
    Core counting logic.
    - Only updates the lid memory buffer when no hand is present
      so occlusion doesn't corrupt the baseline.
    - Calibrates once the buffer is full.
    - On hand-touch: snapshots the current baseline.
    - After hand leaves: waits CONFIRM_THRESHOLD consecutive frames
      where the count stays below the snapshot, then subtracts.
    - On stack appearing (count rises from 0): adds to inventory.
    """

    # ── Update buffer (hand-free frames only) ──
    if not hand_contact:
        s["lid_memory"].append(current_visible)
        if len(s["lid_memory"]) > BUFFER_SIZE:
            s["lid_memory"].pop(0)

    if not s["lid_memory"]:
        return

    # Most common value in buffer = stable count
    stable = max(set(s["lid_memory"]), key=s["lid_memory"].count)

    # ── Calibration ──
    if not s["calibrated"]:
        if len(s["lid_memory"]) == BUFFER_SIZE:
            s["baseline"] = stable
            s["total_inv"] = stable
            s["calibrated"] = True
            s["log"].insert(0, f"{time.strftime('%H:%M:%S')}  Calibrated — {stable} lids")
        return  # don't run event logic until calibrated

    # ── Stack added (tote refilled from empty) ──
    if not hand_contact and stable > s["baseline"] and s["baseline"] == 0:
        added = stable
        s["total_inv"] += added
        s["baseline"] = stable
        s["log"].insert(0, f"{time.strftime('%H:%M:%S')}  Stack Added (+{added})")
        s["log"] = s["log"][:20]

    # ── Hand just touched ──
    if hand_contact and not s["hand_was_present"]:
        s["hand_is_present"] = True
        s["count_at_touch"] = s["baseline"]  # snapshot before anything moves
        s["confirm_frames"] = 0

    # ── Hand just left ──
    if not hand_contact and s["hand_was_present"]:
        s["confirm_frames"] = 0  # start fresh confirmation window

    # ── Confirmation window: count stays lower after hand left ──
    if not hand_contact and s["hand_is_present"]:
        if current_visible < s["count_at_touch"]:
            s["confirm_frames"] += 1
        else:
            # Count recovered — false alarm (camera moved, lid put back, etc.)
            s["confirm_frames"] = 0
            s["hand_is_present"] = False

        if s["confirm_frames"] >= CONFIRM_THRESHOLD:
            removed = s["count_at_touch"] - current_visible
            if removed > 0:
                s["total_inv"] -= removed
                s["baseline"] = current_visible
                s["log"].insert(0, f"{time.strftime('%H:%M:%S')}  Removed (-{removed})")
                s["log"] = s["log"][:20]
            s["hand_is_present"] = False
            s["confirm_frames"] = 0

    # ── Update baseline when hand is clear and count is stable ──
    if not hand_contact and not s["hand_is_present"]:
        s["baseline"] = stable

    s["hand_was_present"] = hand_contact


# ── Render helpers ────────────────────────────────────────
def render_metrics(total_inv, calibrated, cam_status=None):
    cal_badge = (
        "<span class='badge badge-calibrated'>● Calibrated</span>"
        if calibrated else
        "<span class='badge badge-waiting'>◌ Calibrating</span>"
    )
    status_badge = ""
    if cam_status == "ACTIVE":
        status_badge = "<span class='badge badge-active'>● Live</span>"
    elif cam_status == "IDLE":
        status_badge = "<span class='badge badge-idle'>○ Idle</span>"

    metric_placeholder.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Current Inventory</div>
            <div class='metric-value'>{total_inv}</div>
            <div class='metric-sub'>units on production line</div>
            <div>{cal_badge}{status_badge}</div>
        </div>
    """, unsafe_allow_html=True)


def render_log(log):
    if not log:
        log_placeholder.markdown("""
            <div class='log-card'>
                <span style='color:#1e2229'>— awaiting events —</span>
            </div>
        """, unsafe_allow_html=True)
        return

    lines = []
    for entry in log[:8]:
        if "Removed" in entry:
            lines.append(f"<div class='log-entry-remove'>▼ {entry}</div>")
        elif "Added" in entry or "Stack" in entry:
            lines.append(f"<div class='log-entry-add'>▲ {entry}</div>")
        else:
            lines.append(f"<div class='log-entry-cal'>◎ {entry}</div>")

    log_placeholder.markdown(
        f"<div class='log-card'>{''.join(lines)}</div>",
        unsafe_allow_html=True
    )


# ── Video processing loop (demo + upload) ─────────────────
def process_video_loop(cap, frame_window, s, conf):
    frame_count = 0
    last_hands, last_lids = [], []

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Run YOLO every 3rd frame at reduced resolution for speed
        if frame_count % 3 == 0:
            h, w = frame.shape[:2]
            target_w = 320
            target_h = int(target_w * h / w)
            small = cv2.resize(frame, (target_w, target_h))
            results = model(small, conf=conf, imgsz=target_w, verbose=False)
            last_hands, last_lids = [], []
            sx = w / target_w
            sy = h / target_h

            for r in results:
                for box in r.boxes:
                    c = box.xyxy[0].tolist()
                    scaled = [c[0]*sx, c[1]*sy, c[2]*sx, c[3]*sy]
                    label = model.names[int(box.cls[0])]
                    if label == 'hand':
                        last_hands.append(scaled)
                    else:
                        last_lids.append(scaled)

        # Draw last known detections on every frame
        for c in last_hands:
            x1, y1, x2, y2 = map(int, c)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 191, 0), 2)
            cv2.putText(frame, 'hand', (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 191, 0), 1)
        for c in last_lids:
            x1, y1, x2, y2 = map(int, c)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 200, 120), 2)
            cv2.putText(frame, 'lid', (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 200, 120), 1)

        hand_contact = any(
            not (h[2] < l[0] or h[0] > l[2] or h[3] < l[1] or h[1] > l[3])
            for h in last_hands for l in last_lids
        )
        run_logic(len(last_lids), hand_contact, s)

        frame_window.image(frame, channels="BGR", width='stretch')
        render_metrics(s["total_inv"], s["calibrated"])
        render_log(s["log"])
        time.sleep(0.03)


# ── WebRTC processor ──────────────────────────────────────
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
                color = (255, 191, 0) if label == 'hand' else (80, 200, 120)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

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


# ── Layout ────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col2:
    st.markdown("<div class='section-header'>Metrics</div>", unsafe_allow_html=True)
    metric_placeholder = st.empty()
    st.markdown(
        "<div class='section-header' style='margin-top:20px'>Event Log</div>",
        unsafe_allow_html=True
    )
    log_placeholder = st.empty()
    render_metrics(0, False)
    render_log([])

with col1:
    st.markdown("<div class='section-header'>Camera Feed</div>", unsafe_allow_html=True)

    # ── Live Camera ──
    if mode == "Live Camera (WebRTC)":
        for k in ['demo_cap', 'upload_cap']:
            if k in st.session_state:
                st.session_state[k].release()
                del st.session_state[k]
        if 'upload_path' in st.session_state:
            try:
                os.remove(st.session_state.upload_path)
            except Exception:
                pass
            del st.session_state['upload_path']

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
            while True:
                total_inv, log, calibrated = ctx.video_processor.get_display()
                status = "ACTIVE" if ctx.state.playing else "IDLE"
                render_metrics(total_inv, calibrated, cam_status=status)
                render_log(log)
                time.sleep(0.3)
        else:
            render_metrics(0, False, cam_status="IDLE")
            render_log([])

    # ── Demo Video ──
    elif mode == "Demo Video":
        if 'upload_cap' in st.session_state:
            st.session_state.upload_cap.release()
            del st.session_state.upload_cap

        if reset_btn or 's_demo' not in st.session_state:
            st.session_state.s_demo = make_state()
        s_demo = st.session_state.s_demo

        if 'demo_cap' not in st.session_state or reset_btn:
            if 'demo_cap' in st.session_state:
                st.session_state.demo_cap.release()
            st.session_state.demo_cap = cv2.VideoCapture("demo_video.mp4")
            st.session_state.demo_running = False

        frame_window = st.empty()
        c1, c2 = st.columns(2)
        with c1:
            if st.button("▶  Start Demo"):
                st.session_state.demo_running = True
        with c2:
            if st.button("⏹  Stop Demo"):
                st.session_state.demo_running = False

        if st.session_state.get('demo_running', False):
            process_video_loop(st.session_state.demo_cap, frame_window, s_demo, conf_threshold)
        else:
            render_metrics(s_demo["total_inv"], s_demo["calibrated"])
            render_log(s_demo["log"])
            st.markdown("""
                <div style='color:#2d3340;font-size:12px;text-align:center;
                            padding:48px;border:1px dashed #1e2229;border-radius:8px;
                            letter-spacing:1px'>
                    Press ▶ Start Demo to begin
                </div>
            """, unsafe_allow_html=True)

    # ── Upload Video ──
    else:
        if 'demo_cap' in st.session_state:
            st.session_state.demo_cap.release()
            del st.session_state.demo_cap

        if reset_btn or 's_upload' not in st.session_state:
            st.session_state.s_upload = make_state()
        s_upload = st.session_state.s_upload

        uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

        if uploaded is not None:
            if st.session_state.get('upload_name') != uploaded.name:
                if 'upload_cap' in st.session_state:
                    st.session_state.upload_cap.release()
                if 'upload_path' in st.session_state:
                    try:
                        os.remove(st.session_state.upload_path)
                    except Exception:
                        pass
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded.read())
                tfile.flush()
                tfile.close()
                st.session_state.upload_cap = cv2.VideoCapture(tfile.name)
                st.session_state.upload_path = tfile.name
                st.session_state.upload_name = uploaded.name
                st.session_state.upload_running = False
                st.session_state.s_upload = make_state()
                s_upload = st.session_state.s_upload

            frame_window_up = st.empty()
            c1, c2 = st.columns(2)
            with c1:
                if st.button("▶  Start", key="up_start"):
                    st.session_state.upload_running = True
            with c2:
                if st.button("⏹  Stop", key="up_stop"):
                    st.session_state.upload_running = False

            if st.session_state.get('upload_running', False):
                process_video_loop(
                    st.session_state.upload_cap,
                    frame_window_up,
                    s_upload,
                    conf_threshold,
                )
            else:
                render_metrics(s_upload["total_inv"], s_upload["calibrated"])
                render_log(s_upload["log"])
                st.markdown("""
                    <div style='color:#2d3340;font-size:12px;text-align:center;
                                padding:48px;border:1px dashed #1e2229;border-radius:8px;
                                letter-spacing:1px'>
                        Press ▶ Start to begin
                    </div>
                """, unsafe_allow_html=True)
        else:
            render_metrics(0, False)
            render_log([])
            st.markdown("""
                <div style='color:#2d3340;font-size:12px;text-align:center;
                            padding:48px;border:1px dashed #1e2229;border-radius:8px;
                            letter-spacing:1px'>
                    Upload a .mp4 / .mov / .avi to begin
                </div>
            """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────
st.markdown("""
    <div class='footer'>
        <div class='footer-left'>
            Built by&nbsp;
            <strong>Anirudh Yuvaraj</strong>
            &nbsp;&amp;&nbsp;
            <strong>Jonathan Philip</strong>
            &nbsp;·&nbsp;
            <a href='https://github.com/aniwhy/all-CLAD26-ASY-LidCounter/tree/main'
               target='_blank'>GitHub ↗</a>
        </div>
        <div class='footer-right'>All-Clad · 2026</div>
    </div>
""", unsafe_allow_html=True)
