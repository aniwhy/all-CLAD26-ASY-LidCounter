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
    /* ── Base ── */
    .main { background-color: #0a0e14; }
    [data-testid="stAppViewContainer"] { background-color: #0a0e14; }
    [data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #21262d; }
    h1, h2, h3, p, label, .stMarkdown { color: #e6edf3 !important; }

    /* ── Metric cards ── */
    .metric-card {
        background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    }
    .metric-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #8b949e !important;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 42px;
        font-weight: 700;
        color: #f0f6fc !important;
        line-height: 1;
        font-family: 'Courier New', monospace;
    }
    .metric-value-small {
        font-size: 18px;
        font-weight: 600;
        color: #f0f6fc !important;
        font-family: 'Courier New', monospace;
    }

    /* ── Status badges ── */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        margin-top: 8px;
    }
    .badge-active {
        background-color: rgba(35, 197, 94, 0.15);
        color: #23c55e;
        border: 1px solid rgba(35, 197, 94, 0.3);
    }
    .badge-idle {
        background-color: rgba(139, 148, 158, 0.15);
        color: #8b949e;
        border: 1px solid rgba(139, 148, 158, 0.3);
    }
    .badge-calibrated {
        background-color: rgba(88, 166, 255, 0.15);
        color: #58a6ff;
        border: 1px solid rgba(88, 166, 255, 0.3);
    }
    .badge-waiting {
        background-color: rgba(210, 153, 34, 0.15);
        color: #d29922;
        border: 1px solid rgba(210, 153, 34, 0.3);
    }

    /* ── Log card ── */
    .log-card {
        background-color: #0d1117;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 12px 16px;
        font-family: 'Courier New', monospace;
        font-size: 12px;
        color: #8b949e;
        min-height: 160px;
    }
    .log-entry-remove { color: #f85149; }
    .log-entry-add    { color: #23c55e; }
    .log-entry-cal    { color: #58a6ff; }

    /* ── Feed container ── */
    .feed-container {
        background-color: #0d1117;
        border: 1px solid #21262d;
        border-radius: 12px;
        padding: 4px;
        overflow: hidden;
    }

    /* ── Section headers ── */
    .section-header {
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #58a6ff !important;
        border-bottom: 1px solid #21262d;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }

    /* ── Buttons ── */
    .stButton > button {
        background-color: #21262d;
        color: #e6edf3;
        border: 1px solid #30363d;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background-color: #30363d;
        border-color: #58a6ff;
        color: #58a6ff;
    }

    /* ── Slider ── */
    .stSlider > div > div > div > div { background-color: #58a6ff; }

    /* ── Hide default streamlit metric styling ── */
    [data-testid="stMetric"] { display: none; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0d1117; }
    ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
    </style>
    """, unsafe_allow_html=True)


# ── Header ──────────────────────────────────────────────
col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.image("logo.png", width=80)
with col_title:
    st.markdown("""
        <div style='padding-top:8px'>
            <div style='font-size:24px;font-weight:700;color:#f0f6fc;letter-spacing:1px'>
                ALL-CLAD LID INVENTORY
            </div>
            <div style='font-size:12px;color:#8b949e;letter-spacing:2px;text-transform:uppercase'>
                Computer Vision Tracking System · Anirudh Yuvaraj
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:1px;background:#21262d;margin:12px 0 20px 0'></div>",
            unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────
st.sidebar.markdown("""
    <div class='section-header'>System Control</div>
""", unsafe_allow_html=True)

conf_threshold = st.sidebar.slider("Detection Confidence", 0.3, 0.9, 0.5)
reset_btn = st.sidebar.button("⟳  Hard Reset")
mode = st.sidebar.radio("Input Mode", ["Live Camera (WebRTC)", "Demo Video", "Upload Video"])

st.sidebar.markdown("<div style='height:1px;background:#21262d;margin:16px 0'></div>",
                    unsafe_allow_html=True)
st.sidebar.markdown("""
    <div style='font-size:10px;color:#8b949e;letter-spacing:1px;text-transform:uppercase'>
        Model: lidDetection.pt<br>
        Buffer: 15 frames<br>
        Confirm threshold: 8
    </div>
""", unsafe_allow_html=True)


# ── Model ────────────────────────────────────────────────
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
        s["log"].insert(0, f"{time.strftime('%H:%M:%S')} ·  Calibrated ({stable_count} lids)")
        return

    if not s["calibrated"]:
        return

    if (not hand_contact
            and stable_count > s["last_stable_count"]
            and s["last_stable_count"] == 0):
        s["total_inv"] += stable_count
        s["log"].insert(0, f"{time.strftime('%H:%M:%S')} ·  Stack Added (+{stable_count})")
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
                s["log"].insert(0, f"{time.strftime('%H:%M:%S')} ·  Removed (-{removed})")
                s["last_stable_count"] = current_visible
                s["log"] = s["log"][:20]
            s["touching"] = False
            s["confirm"] = 0

    s["prev_hand"] = hand_contact

    if not hand_contact and not s["touching"]:
        s["last_stable_count"] = stable_count


def render_metrics(total_inv, calibrated, cam_status=None):
    cal_badge = (
        "<span class='badge badge-calibrated'>● Calibrated</span>"
        if calibrated else
        "<span class='badge badge-waiting'>◌ Calibrating...</span>"
    )
    status_badge = ""
    if cam_status == "ACTIVE":
        status_badge = "<span class='badge badge-active'>● Live</span>"
    elif cam_status == "IDLE":
        status_badge = "<span class='badge badge-idle'>● Idle</span>"

    metric_placeholder.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Current Inventory</div>
            <div class='metric-value'>{total_inv}</div>
            <div style='margin-top:4px;font-size:13px;color:#8b949e'>units on line</div>
            <div style='margin-top:10px'>{cal_badge} {status_badge}</div>
        </div>
    """, unsafe_allow_html=True)


def render_log(log):
    if not log:
        log_placeholder.markdown("""
            <div class='log-card'>
                <span style='color:#3d444d'>No events yet...</span>
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


def process_video_loop(cap, frame_window, s, conf):
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf, imgsz=640, verbose=False)
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
        render_metrics(s["total_inv"], s["calibrated"])
        render_log(s["log"])
        time.sleep(0.05)


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


# ── Layout ───────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col2:
    st.markdown("<div class='section-header'>Metrics</div>", unsafe_allow_html=True)
    metric_placeholder = st.empty()
    st.markdown("<div class='section-header' style='margin-top:20px'>Event Log</div>",
                unsafe_allow_html=True)
    log_placeholder = st.empty()

    # Initial empty state
    render_metrics(0, False)
    render_log([])

with col1:
    st.markdown("<div class='section-header'>Camera Feed</div>", unsafe_allow_html=True)

    if mode == "Live Camera (WebRTC)":
        for cap_key in ['demo_cap', 'upload_cap']:
            if cap_key in st.session_state:
                st.session_state[cap_key].release()
                del st.session_state[cap_key]
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
        col_start, col_stop = st.columns(2)
        with col_start:
            if st.button("▶  Start Demo"):
                st.session_state.demo_running = True
        with col_stop:
            if st.button("⏹  Stop Demo"):
                st.session_state.demo_running = False

        if st.session_state.get('demo_running', False):
            process_video_loop(
                st.session_state.demo_cap,
                frame_window,
                s_demo,
                conf_threshold,
            )
        else:
            render_metrics(s_demo["total_inv"], s_demo["calibrated"])
            render_log(s_demo["log"])
            st.markdown("""
                <div style='color:#3d444d;font-size:13px;text-align:center;
                            padding:40px;border:1px dashed #21262d;border-radius:8px'>
                    Press ▶ Start Demo to begin
                </div>
            """, unsafe_allow_html=True)

    else:  # Upload Video
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
            col_start_u, col_stop_u = st.columns(2)
            with col_start_u:
                if st.button("▶  Start", key="up_start"):
                    st.session_state.upload_running = True
            with col_stop_u:
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
                    <div style='color:#3d444d;font-size:13px;text-align:center;
                                padding:40px;border:1px dashed #21262d;border-radius:8px'>
                        Press ▶ Start to begin
                    </div>
                """, unsafe_allow_html=True)
        else:
            render_metrics(0, False)
            render_log([])
            st.markdown("""
                <div style='color:#3d444d;font-size:13px;text-align:center;
                            padding:40px;border:1px dashed #21262d;border-radius:8px'>
                    Upload a video file to begin
                </div>
            """, unsafe_allow_html=True)
