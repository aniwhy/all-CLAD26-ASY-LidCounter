import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
from ultralytics import YOLO
import time
import av
import threading
import tempfile
import os

st.set_page_config(page_title="All-Clad Lid Inventory", layout="wide", initial_sidebar_state="collapsed")

# ── Theme state ───────────────────────────────────────────
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True
if 'show_app' not in st.session_state:
    st.session_state.show_app = False

dark = st.session_state.dark_mode

# ── Theme variables ───────────────────────────────────────
if dark:
    BG          = "#111318"
    BG2         = "#0d0f13"
    BG3         = "#16191f"
    BORDER      = "#252830"
    TEXT        = "#d4cfc8"
    TEXT_DIM    = "#8a8278"
    TEXT_DIMMER = "#3d4048"
    BEIGE       = "#c8b89a"
    BEIGE_DIM   = "#8a7d68"
    RED         = "#c41230"
    RED_BRIGHT  = "#e8394f"
    ARROW       = "#c8b89a"
else:
    BG          = "#f5f0eb"
    BG2         = "#ede8e0"
    BG3         = "#e8e0d5"
    BORDER      = "#c8bfb0"
    TEXT        = "#1a1814"
    TEXT_DIM    = "#5a5248"
    TEXT_DIMMER = "#8a8278"
    BEIGE       = "#7a6a54"
    BEIGE_DIM   = "#5a4a38"
    RED         = "#c41230"
    RED_BRIGHT  = "#a00e28"
    ARROW       = "#7a6a54"

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"], [data-testid],
    *, *::before, *::after,
    button, input, select, textarea, p, span, div, a, label {{
        font-family: 'Inter', sans-serif !important;
    }}

    .main, [data-testid="stAppViewContainer"] {{
        background-color: {BG};
    }}
    [data-testid="stSidebar"] {{
        background-color: {BG2};
        border-right: 1px solid {BORDER};
    }}
    p, span, div, label, .stMarkdown,
    [data-testid="stText"],
    [data-testid="stMarkdownContainer"] p {{
        color: {TEXT} !important;
    }}
    h1, h2, h3 {{ color: {TEXT} !important; }}

    /* Welcome screen */
    .welcome-screen {{
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: {BG2};
        z-index: 9999;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 40px;
    }}
    .welcome-logo {{
        width: 120px;
        margin-bottom: 24px;
        opacity: 0.95;
    }}
    .welcome-title {{
        font-size: 32px;
        font-weight: 700;
        color: {TEXT};
        letter-spacing: 1px;
        margin-bottom: 8px;
    }}
    .welcome-sub {{
        font-size: 13px;
        color: {TEXT_DIM};
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 48px;
    }}
    .welcome-divider {{
        width: 48px;
        height: 2px;
        background: {RED};
        margin: 0 auto 48px auto;
    }}
    .welcome-hint {{
        font-size: 12px;
        color: {TEXT_DIMMER};
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-top: 4px;
    }}
    .welcome-arrow {{
        font-size: 22px;
        color: {ARROW};
        animation: bounce 1.8s ease infinite;
        display: block;
        margin-top: 12px;
    }}
    @keyframes bounce {{
        0%, 100% {{ transform: translateY(0); opacity: 0.6; }}
        50%       {{ transform: translateY(8px); opacity: 1; }}
    }}
    @keyframes fadeOut {{
        from {{ opacity: 1; transform: translateY(0); }}
        to   {{ opacity: 0; transform: translateY(-40px); }}
    }}
    .welcome-screen.hiding {{
        animation: fadeOut 0.5s ease forwards;
    }}

    /* Animations */
    @keyframes pulse-red {{
        0%   {{ box-shadow: 0 0 0 0 rgba(196,18,48,0.5); }}
        70%  {{ box-shadow: 0 0 0 8px rgba(196,18,48,0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(196,18,48,0); }}
    }}
    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(5px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes shimmer-red {{
        0%   {{ background-position: 0%; }}
        100% {{ background-position: 200%; }}
    }}
    @keyframes blink {{
        0%, 100% {{ opacity: 1; }}
        50%       {{ opacity: 0.4; }}
    }}

    /* Metric card */
    .metric-card {{
        background: {BG3};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 22px 24px;
        margin-bottom: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        animation: fadeInUp 0.3s ease;
        position: relative;
        overflow: hidden;
    }}
    .metric-card::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, #6b0015, {RED}, {RED_BRIGHT}, {RED}, #6b0015);
        background-size: 200%;
        animation: shimmer-red 4s linear infinite;
    }}
    .metric-label {{
        font-size: 10px !important;
        font-weight: 700 !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        color: {TEXT_DIM} !important;
        margin-bottom: 10px !important;
    }}
    .metric-value {{
        font-size: 58px !important;
        font-weight: 700 !important;
        color: {TEXT} !important;
        line-height: 1 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }}
    .metric-sub {{
        font-size: 11px !important;
        color: {TEXT_DIMMER} !important;
        margin-top: 6px !important;
        letter-spacing: 1px !important;
    }}

    /* Badges */
    .badge {{
        display: inline-block;
        padding: 4px 11px;
        border-radius: 20px;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-top: 12px;
        margin-right: 5px;
    }}
    .badge-active {{
        background: rgba(196,18,48,0.12);
        color: {RED_BRIGHT};
        border: 1px solid rgba(196,18,48,0.35);
        animation: pulse-red 2s infinite;
    }}
    .badge-idle {{
        background: rgba(138,130,120,0.1);
        color: {TEXT_DIM};
        border: 1px solid rgba(138,130,120,0.2);
    }}
    .badge-calibrated {{
        background: rgba(200,184,154,0.1);
        color: {BEIGE};
        border: 1px solid rgba(200,184,154,0.25);
    }}
    .badge-waiting {{
        background: rgba(196,18,48,0.08);
        color: {RED};
        border: 1px solid rgba(196,18,48,0.2);
        animation: blink 1.5s ease infinite;
    }}

    /* Log card */
    .log-card {{
        background: {BG2};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 14px 16px;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 11px !important;
        color: {TEXT_DIM};
        min-height: 200px;
        line-height: 2.2;
    }}
    .log-entry-remove {{
        color: {RED_BRIGHT} !important;
        font-family: 'JetBrains Mono', monospace !important;
    }}
    .log-entry-add {{
        color: {BEIGE} !important;
        font-family: 'JetBrains Mono', monospace !important;
    }}
    .log-entry-cal {{
        color: {TEXT_DIMMER} !important;
        font-family: 'JetBrains Mono', monospace !important;
    }}

    /* Section headers */
    .section-header {{
        font-size: 10px !important;
        font-weight: 700 !important;
        letter-spacing: 2.5px !important;
        text-transform: uppercase !important;
        color: {RED} !important;
        border-bottom: 1px solid {BORDER} !important;
        padding-bottom: 8px !important;
        margin-bottom: 16px !important;
    }}

    /* Buttons */
    button,
    .stButton > button,
    [data-testid="baseButton-primary"],
    [data-testid="baseButton-secondary"],
    [data-baseweb="button"] {{
        font-family: 'Inter', sans-serif !important;
        background-color: {BG3} !important;
        color: {BEIGE} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 6px !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        letter-spacing: 0.3px !important;
        transition: all 0.2s !important;
        padding: 8px 16px !important;
    }}
    button:hover,
    .stButton > button:hover,
    [data-testid="baseButton-primary"]:hover,
    [data-testid="baseButton-secondary"]:hover,
    [data-baseweb="button"]:hover {{
        border-color: {RED} !important;
        color: {RED_BRIGHT} !important;
        background-color: {BG2} !important;
        box-shadow: 0 0 10px rgba(196,18,48,0.12) !important;
    }}

    /* Slider */
    .stSlider > div > div > div > div {{
        background-color: {RED} !important;
    }}
    .stSlider label {{
        color: {TEXT_DIM} !important;
        font-size: 12px !important;
    }}

    /* Radio */
    .stRadio > label {{
        color: {TEXT_DIM} !important;
        font-size: 11px !important;
    }}
    .stRadio [data-testid="stMarkdownContainer"] p {{
        color: {TEXT} !important;
        font-size: 13px !important;
    }}

    /* File uploader */
    [data-testid="stFileUploader"] {{
        background: {BG2} !important;
        border: 1px dashed {BORDER} !important;
        border-radius: 8px !important;
    }}
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] p {{
        color: {TEXT_DIM} !important;
        font-size: 12px !important;
    }}

    /* Hide default metric */
    [data-testid="stMetric"] {{ display: none; }}

    /* Sidebar text */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div {{
        color: {TEXT_DIM} !important;
        font-size: 12px !important;
    }}

    /* Theme toggle button — special styling */
    .theme-btn > button {{
        background: transparent !important;
        border: 1px solid {BORDER} !important;
        color: {BEIGE} !important;
        font-size: 13px !important;
        padding: 6px 14px !important;
        border-radius: 20px !important;
    }}
    .theme-btn > button:hover {{
        border-color: {RED} !important;
        color: {RED_BRIGHT} !important;
    }}

    /* Footer */
    .footer {{
        margin-top: 48px;
        padding: 18px 0 8px 0;
        border-top: 1px solid {BORDER};
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 10px;
    }}
    .footer-left {{
        font-size: 12px;
        color: {TEXT_DIMMER};
    }}
    .footer-left strong {{ color: {TEXT_DIM}; font-weight: 600; }}
    .footer-right {{
        font-size: 10px;
        color: {TEXT_DIMMER};
        letter-spacing: 2px;
        text-transform: uppercase;
    }}
    .footer a {{ color: {RED} !important; text-decoration: none; }}
    .footer a:hover {{ color: {RED_BRIGHT} !important; }}

    /* Scrollbar */
    ::-webkit-scrollbar {{ width: 4px; }}
    ::-webkit-scrollbar-track {{ background: {BG2}; }}
    ::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 2px; }}
    </style>
""", unsafe_allow_html=True)


# ── Welcome screen ────────────────────────────────────────
if not st.session_state.show_app:
    st.markdown(f"""
        <div class="welcome-screen" id="welcome">
            <img src="app/static/logo.png" class="welcome-logo"
                 onerror="this.style.display='none'"/>
            <div class="welcome-title">All-Clad Lid Inventory</div>
            <div class="welcome-sub">Computer Vision Tracking System</div>
            <div class="welcome-divider"></div>
            <div class="welcome-hint">Scroll down or tap to begin</div>
            <span class="welcome-arrow">↓</span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <script>
        // Listen for scroll or click to dismiss welcome screen
        function dismissWelcome() {{
            var el = document.getElementById('welcome');
            if (el) {{
                el.classList.add('hiding');
                setTimeout(function() {{ el.style.display = 'none'; }}, 500);
            }}
        }}
        window.addEventListener('scroll', dismissWelcome, {{ once: true }});
        document.addEventListener('click', dismissWelcome, {{ once: true }});
        </script>
    """, unsafe_allow_html=True)

    col_enter = st.columns([3, 2, 3])[1]
    with col_enter:
        st.markdown("<div style='height:60vh'></div>", unsafe_allow_html=True)
        if st.button("Enter →", key="enter_btn"):
            st.session_state.show_app = True
            st.rerun()
    st.stop()


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
        "lid_memory": [],
        "baseline": 0,
        "calibrated": False,
        "hand_was_present": False,
        "hand_is_present": False,
        "count_at_touch": 0,
        "confirm_frames": 0,
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

    stable = max(set(s["lid_memory"]), key=s["lid_memory"].count)

    if not s["calibrated"]:
        if len(s["lid_memory"]) == BUFFER_SIZE:
            s["baseline"] = stable
            s["total_inv"] = stable
            s["calibrated"] = True
            s["log"].insert(0, f"{time.strftime('%H:%M:%S')}  Calibrated — {stable} lids")
        return

    # Stack added
    if not hand_contact and stable > s["baseline"] and s["baseline"] == 0:
        s["total_inv"] += stable
        s["baseline"] = stable
        s["log"].insert(0, f"{time.strftime('%H:%M:%S')}  Stack Added (+{stable})")
        s["log"] = s["log"][:20]

    # Hand just touched
    if hand_contact and not s["hand_was_present"]:
        s["hand_is_present"] = True
        s["count_at_touch"] = s["baseline"]
        s["confirm_frames"] = 0

    # Hand just left
    if not hand_contact and s["hand_was_present"]:
        s["confirm_frames"] = 0

    # Confirmation window
    if not hand_contact and s["hand_is_present"]:
        if current_visible < s["count_at_touch"]:
            s["confirm_frames"] += 1
        else:
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
        log_placeholder.markdown(f"""
            <div class='log-card'>
                <span style='color:{BORDER};font-family:JetBrains Mono,monospace'>
                    — awaiting events —
                </span>
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

        if frame_count % 3 == 0:
            h, w = frame.shape[:2]
            tw, th = 320, int(320 * h / w)
            small = cv2.resize(frame, (tw, th))
            results = model(small, conf=conf, imgsz=tw, verbose=False)
            last_hands, last_lids = [], []
            sx, sy = w / tw, h / th
            for r in results:
                for box in r.boxes:
                    c = box.xyxy[0].tolist()
                    scaled = [c[0]*sx, c[1]*sy, c[2]*sx, c[3]*sy]
                    label = model.names[int(box.cls[0])]
                    if label == 'hand':
                        last_hands.append(scaled)
                    else:
                        last_lids.append(scaled)

        for c in last_hands:
            x1, y1, x2, y2 = map(int, c)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 170, 120), 2)
            cv2.putText(frame, 'hand', (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 170, 120), 1)
        for c in last_lids:
            x1, y1, x2, y2 = map(int, c)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (48, 18, 196), 2)
            cv2.putText(frame, 'lid', (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (48, 18, 196), 1)

        hand_contact = any(
            not (h[2] < l[0] or h[0] > l[2] or h[3] < l[1] or h[1] > l[3])
            for h in last_hands for l in last_lids
        )
        run_logic(len(last_lids), hand_contact, s)
        frame_window.image(frame, channels="BGR", width='stretch')
        render_metrics(s["total_inv"], s["calibrated"])
        render_log(s["log"])
        time.sleep(0.03)


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
                color = (200, 170, 120) if label == 'hand' else (48, 18, 196)
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


# ── Header ────────────────────────────────────────────────
header_l, header_r = st.columns([6, 1])
with header_l:
    col_logo, col_title = st.columns([1, 8])
    with col_logo:
        st.image("logo.png", width=68)
    with col_title:
        st.markdown(f"""
            <div style='padding-top:10px'>
                <div style='font-size:22px;font-weight:700;color:{TEXT};letter-spacing:0.5px'>
                    All-Clad Lid Inventory
                </div>
                <div style='font-size:11px;color:{TEXT_DIMMER};letter-spacing:2px;
                            text-transform:uppercase;margin-top:4px'>
                    Computer Vision Tracking &nbsp;·&nbsp; Production Line 1
                </div>
            </div>
        """, unsafe_allow_html=True)

with header_r:
    st.markdown("<div style='padding-top:14px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='theme-btn'>", unsafe_allow_html=True)
    toggle_label = "☀ Light" if dark else "☾ Dark"
    if st.button(toggle_label, key="theme_toggle"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    f"<div style='height:1px;background:linear-gradient(90deg,transparent,{RED},transparent);"
    f"margin:14px 0 22px 0'></div>",
    unsafe_allow_html=True
)

# ── Sidebar ───────────────────────────────────────────────
st.sidebar.markdown(
    "<div class='section-header'>System Control</div>",
    unsafe_allow_html=True
)
conf_threshold = st.sidebar.slider("Detection Confidence", 0.3, 0.9, 0.5)
reset_btn = st.sidebar.button("⟳  Hard Reset")
mode = st.sidebar.radio("Input Mode", ["Live Camera (WebRTC)", "Demo Video", "Upload Video"])
st.sidebar.markdown(
    f"<div style='height:1px;background:{BORDER};margin:16px 0'></div>",
    unsafe_allow_html=True
)
st.sidebar.markdown(f"""
    <div style='font-size:10px;color:{TEXT_DIMMER};letter-spacing:1px;
                text-transform:uppercase;line-height:2.4'>
        Model · lidDetection.pt<br>
        Buffer · 15 frames<br>
        Confirm · 8 frames
    </div>
""", unsafe_allow_html=True)


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
            st.markdown(f"""
                <div style='color:{TEXT_DIMMER};font-size:12px;text-align:center;
                            padding:48px;border:1px dashed {BORDER};border-radius:8px;
                            letter-spacing:1px'>
                    Press ▶ Start Demo to begin
                </div>
            """, unsafe_allow_html=True)

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
                st.markdown(f"""
                    <div style='color:{TEXT_DIMMER};font-size:12px;text-align:center;
                                padding:48px;border:1px dashed {BORDER};border-radius:8px;
                                letter-spacing:1px'>
                        Press ▶ Start to begin
                    </div>
                """, unsafe_allow_html=True)
        else:
            render_metrics(0, False)
            render_log([])
            st.markdown(f"""
                <div style='color:{TEXT_DIMMER};font-size:12px;text-align:center;
                            padding:48px;border:1px dashed {BORDER};border-radius:8px;
                            letter-spacing:1px'>
                    Upload a .mp4 / .mov / .avi to begin
                </div>
            """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────
st.markdown(f"""
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
