import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
from ultralytics import YOLO
import time
import av
import threading
import tempfile
import os
import base64

st.set_page_config(
    page_title="All-Clad Lid Inventory",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── State ─────────────────────────────────────────────────
if 'dark_mode' not in st.session_state: st.session_state.dark_mode = True
if 'show_app'  not in st.session_state: st.session_state.show_app  = False

dark = st.session_state.dark_mode

# ── Theme ─────────────────────────────────────────────────
if dark:
    BG          = "#111318"
    BG2         = "#0d0f13"
    BG3         = "#16191f"
    BORDER      = "#252830"
    TEXT        = "#d4cfc8"
    TEXT_DIM    = "#8a8278"
    TEXT_DIMMER = "#3d4048"
    BEIGE       = "#c8b89a"
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
    RED         = "#c41230"
    RED_BRIGHT  = "#a00e28"
    ARROW       = "#7a6a54"

# ── Logo ──────────────────────────────────────────────────
logo_b64 = ""
logo_html = ""
try:
    with open("logo.png", "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode()
    logo_html = (
        f"<img src='data:image/png;base64,{logo_b64}' "
        f"style='width:90px;margin-bottom:32px;opacity:0.95;'/>"
    )
except Exception:
    pass

# ── CSS ───────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&family=Playfair+Display:wght@400;500;600&display=swap');

html, body, [class*="css"], [data-testid],
*, *::before, *::after,
button, input, select, textarea, p, span, div, a, label {{
    font-family: 'Inter', sans-serif !important;
}}
.main, [data-testid="stAppViewContainer"] {{
    background-color: {BG} !important;
}}
[data-testid="stSidebar"] {{
    background-color: {BG2} !important;
    border-right: 1px solid {BORDER} !important;
}}
[data-testid="collapsedControl"] {{ display: none !important; }}
p, span, label, .stMarkdown,
[data-testid="stText"],
[data-testid="stMarkdownContainer"] p {{ color: {TEXT} !important; }}
h1, h2, h3 {{ color: {TEXT} !important; }}

@keyframes pulse-red {{
    0%   {{ box-shadow: 0 0 0 0 rgba(196,18,48,0.5); }}
    70%  {{ box-shadow: 0 0 0 8px rgba(196,18,48,0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(196,18,48,0); }}
}}
@keyframes fadeInUp {{
    from {{ opacity: 0; transform: translateY(6px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes fadeIn {{
    from {{ opacity: 0; }}
    to   {{ opacity: 1; }}
}}
@keyframes shimmer-red {{
    0%   {{ background-position: 0%; }}
    100% {{ background-position: 200%; }}
}}
@keyframes blink {{
    0%, 100% {{ opacity: 1; }}
    50%       {{ opacity: 0.4; }}
}}
@keyframes bounce {{
    0%, 100% {{ transform: translateY(0);   opacity: 0.4; }}
    50%       {{ transform: translateY(9px); opacity: 1;   }}
}}

.welcome-wrap {{
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; min-height: 92vh;
    text-align: center; padding: 40px 20px 20px 20px;
    animation: fadeIn 0.6s ease;
}}
.welcome-eyebrow {{
    font-size: 10px; font-weight: 600; letter-spacing: 4px;
    text-transform: uppercase; color: {RED}; margin-bottom: 18px;
}}
.welcome-title {{
    font-family: 'Playfair Display', serif !important;
    font-size: 48px; font-weight: 500; color: {TEXT};
    line-height: 1.1; margin-bottom: 12px;
}}
.welcome-sub {{
    font-size: 13px; font-weight: 400; color: {TEXT_DIM};
    letter-spacing: 1.5px; text-transform: uppercase;
    margin-bottom: 44px; line-height: 1.6;
}}
.welcome-divider {{
    width: 32px; height: 1px; background: {RED};
    margin: 0 auto 44px auto; opacity: 0.6;
}}
.welcome-hint {{
    font-size: 10px; font-weight: 500; color: {TEXT_DIMMER};
    letter-spacing: 2.5px; text-transform: uppercase; margin-bottom: 6px;
}}
.welcome-arrow {{
    font-size: 18px; color: {ARROW};
    animation: bounce 2s ease infinite;
    display: block; margin-top: 4px; margin-bottom: 32px;
}}

.metric-card {{
    background: {BG3}; border: 1px solid {BORDER};
    border-radius: 10px; padding: 22px 24px; margin-bottom: 12px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    animation: fadeInUp 0.3s ease; position: relative; overflow: hidden;
}}
.metric-card::before {{
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg,#6b0015,{RED},{RED_BRIGHT},{RED},#6b0015);
    background-size: 200%; animation: shimmer-red 4s linear infinite;
}}
.metric-label {{
    font-size: 10px !important; font-weight: 700 !important;
    letter-spacing: 2px !important; text-transform: uppercase !important;
    color: {TEXT_DIM} !important; margin-bottom: 10px !important;
}}
.metric-value {{
    font-size: 58px !important; font-weight: 700 !important;
    color: {TEXT} !important; line-height: 1 !important;
    font-family: 'JetBrains Mono', monospace !important;
}}
.metric-sub {{
    font-size: 11px !important; color: {TEXT_DIMMER} !important;
    margin-top: 6px !important; letter-spacing: 1px !important;
}}

.badge {{
    display: inline-block; padding: 4px 11px; border-radius: 20px;
    font-size: 10px; font-weight: 700; letter-spacing: 1px;
    text-transform: uppercase; margin-top: 12px; margin-right: 5px;
}}
.badge-active {{
    background: rgba(196,18,48,0.12); color: {RED_BRIGHT};
    border: 1px solid rgba(196,18,48,0.35); animation: pulse-red 2s infinite;
}}
.badge-idle {{
    background: rgba(138,130,120,0.1); color: {TEXT_DIM};
    border: 1px solid rgba(138,130,120,0.2);
}}
.badge-calibrated {{
    background: rgba(200,184,154,0.1); color: {BEIGE};
    border: 1px solid rgba(200,184,154,0.25);
}}
.badge-waiting {{
    background: rgba(196,18,48,0.08); color: {RED};
    border: 1px solid rgba(196,18,48,0.2); animation: blink 1.5s ease infinite;
}}

.log-card {{
    background: {BG2}; border: 1px solid {BORDER}; border-radius: 8px;
    padding: 14px 16px; font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important; color: {TEXT_DIM}; min-height: 200px; line-height: 2.2;
}}
.log-entry-remove {{ color: {RED_BRIGHT} !important; font-family: 'JetBrains Mono', monospace !important; }}
.log-entry-add    {{ color: {BEIGE} !important;      font-family: 'JetBrains Mono', monospace !important; }}
.log-entry-cal    {{ color: {TEXT_DIMMER} !important; font-family: 'JetBrains Mono', monospace !important; }}

.section-header {{
    font-size: 10px !important; font-weight: 700 !important;
    letter-spacing: 2.5px !important; text-transform: uppercase !important;
    color: {RED} !important; border-bottom: 1px solid {BORDER} !important;
    padding-bottom: 8px !important; margin-bottom: 16px !important;
}}

button, .stButton > button,
[data-testid="baseButton-primary"],
[data-testid="baseButton-secondary"],
[data-baseweb="button"] {{
    font-family: 'Inter', sans-serif !important;
    background-color: {BG3} !important; color: {BEIGE} !important;
    border: 1px solid {BORDER} !important; border-radius: 6px !important;
    font-size: 12px !important; font-weight: 500 !important;
    letter-spacing: 0.3px !important; transition: all 0.2s !important;
    padding: 8px 16px !important;
}}
button:hover, .stButton > button:hover,
[data-testid="baseButton-primary"]:hover,
[data-testid="baseButton-secondary"]:hover,
[data-baseweb="button"]:hover {{
    border-color: {RED} !important; color: {RED_BRIGHT} !important;
    background-color: {BG2} !important;
    box-shadow: 0 0 10px rgba(196,18,48,0.12) !important;
}}
.theme-btn > button {{
    background: transparent !important; border: 1px solid {BORDER} !important;
    color: {BEIGE} !important; font-size: 12px !important;
    padding: 6px 14px !important; border-radius: 20px !important;
    letter-spacing: 0.5px !important;
}}
.theme-btn > button:hover {{
    border-color: {RED} !important; color: {RED_BRIGHT} !important;
    background: transparent !important; box-shadow: none !important;
}}

.stSlider > div > div > div > div {{ background-color: {RED} !important; }}
.stSlider label {{ color: {TEXT_DIM} !important; font-size: 12px !important; }}
.stRadio > label {{ color: {TEXT_DIM} !important; font-size: 11px !important; }}
.stRadio [data-testid="stMarkdownContainer"] p {{
    color: {TEXT} !important; font-size: 13px !important;
}}
[data-testid="stFileUploader"] {{
    background: {BG2} !important; border: 1px dashed {BORDER} !important;
    border-radius: 8px !important;
}}
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p {{
    color: {TEXT_DIM} !important; font-size: 12px !important;
}}
[data-testid="stMetric"] {{ display: none; }}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {{
    color: {TEXT_DIM} !important; font-size: 12px !important;
}}

.footer {{
    margin-top: 48px; padding: 18px 0 8px 0;
    border-top: 1px solid {BORDER}; display: flex;
    justify-content: space-between; align-items: center;
    flex-wrap: wrap; gap: 10px;
}}
.footer-left  {{ font-size: 12px; color: {TEXT_DIMMER}; }}
.footer-left strong {{ color: {TEXT_DIM}; font-weight: 600; }}
.footer-right {{ font-size: 10px; color: {TEXT_DIMMER}; letter-spacing: 2px; text-transform: uppercase; }}
.footer a       {{ color: {RED} !important; text-decoration: none; }}
.footer a:hover {{ color: {RED_BRIGHT} !important; }}

.placeholder-box {{
    color: {TEXT_DIMMER}; font-size: 12px; text-align: center;
    padding: 48px; border: 1px dashed {BORDER};
    border-radius: 8px; letter-spacing: 1px;
}}

::-webkit-scrollbar       {{ width: 4px; }}
::-webkit-scrollbar-track {{ background: {BG2}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 2px; }}
</style>
""", unsafe_allow_html=True)


# ── Welcome screen ────────────────────────────────────────
if not st.session_state.show_app:
    st.markdown("""
        <style>
        [data-testid="stSidebar"]        { display: none !important; }
        [data-testid="collapsedControl"] { display: none !important; }
        [data-testid="stHeader"]         { display: none !important; }
        header                           { display: none !important; }
        .main .block-container           { padding-top: 0 !important; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class="welcome-wrap">
            {logo_html}
            <div class="welcome-eyebrow">All-Clad &nbsp;·&nbsp; 2026</div>
            <div class="welcome-title">Lid Inventory<br>Tracking System</div>
            <div class="welcome-sub">Computer Vision &nbsp;·&nbsp; Production Line 1</div>
            <div class="welcome-divider"></div>
            <div class="welcome-hint">Click to enter</div>
            <span class="welcome-arrow">↓</span>
        </div>
    """, unsafe_allow_html=True)

    _, mid, _ = st.columns([2, 1, 2])
    with mid:
        if st.button("Enter →", key="enter_btn"):
            st.session_state.show_app = True
            st.rerun()
    st.stop()


# ── Model ─────────────────────────────────────────────────
@st.cache_resource
def get_model():
    return YOLO('lidDetection.pt')

model     = get_model()
BUFFER_SIZE       = 15
CONFIRM_THRESHOLD = 8

RTC_CONFIG = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
})


# ── Shared counting logic (pure function, no locks needed) ─
def _apply_logic(cv, hc, mem, baseline, calibrated,
                 hand_was, hand_is, count_at, confirm,
                 total_inv, log):
    """
    cv          = current_visible (int)
    hc          = hand_contact (bool)
    Returns updated tuple of all state fields.
    """
    # Only update memory when hand is not in frame
    if not hc:
        mem.append(cv)
        if len(mem) > BUFFER_SIZE:
            mem.pop(0)

    if not mem:
        return mem, baseline, calibrated, hand_was, hand_is, count_at, confirm, total_inv, log

    stable = max(set(mem), key=mem.count)

    # Calibrate
    if not calibrated:
        if len(mem) == BUFFER_SIZE:
            baseline   = stable
            total_inv  = stable
            calibrated = True
            log = [f"{time.strftime('%H:%M:%S')}  Calibrated — {stable} lids"] + log
            log = log[:20]
        return mem, baseline, calibrated, hand_was, hand_is, count_at, confirm, total_inv, log

    # Stack added from empty
    if not hc and stable > baseline and baseline == 0:
        total_inv += stable
        baseline   = stable
        log = [f"{time.strftime('%H:%M:%S')}  Stack Added (+{stable})"] + log
        log = log[:20]

    # Hand just touched
    if hc and not hand_was:
        hand_is   = True
        count_at  = baseline
        confirm   = 0

    # Hand just left
    if not hc and hand_was:
        confirm = 0

    # Confirmation window
    if not hc and hand_is:
        if cv < count_at:
            confirm += 1
        else:
            confirm  = 0
            hand_is  = False

        if confirm >= CONFIRM_THRESHOLD:
            removed = count_at - cv
            if removed > 0:
                total_inv -= removed
                baseline   = cv
                log = [f"{time.strftime('%H:%M:%S')}  Removed (-{removed})"] + log
                log = log[:20]
            hand_is = False
            confirm = 0

    # Update baseline when fully clear
    if not hc and not hand_is:
        baseline = stable

    hand_was = hc

    return mem, baseline, calibrated, hand_was, hand_is, count_at, confirm, total_inv, log


# ── WebRTC processor ──────────────────────────────────────
class LidDetector(VideoProcessorBase):
    """
    All logic runs inside recv() at full camera framerate.
    Private _ fields are only touched by recv() thread.
    Public display fields are protected by a lock for main thread reads.
    """

    def __init__(self):
        self.conf = 0.5

        # Private logic state — recv() thread only, no locking needed
        self._mem        = []
        self._baseline   = 0
        self._cal        = False
        self._hand_was   = False
        self._hand_is    = False
        self._count_at   = 0
        self._confirm    = 0
        self._total      = 0
        self._log        = []

        # Public display state — protected by lock
        self._lock       = threading.Lock()
        self._d_total    = 0
        self._d_log      = []
        self._d_cal      = False

    def reset(self):
        # Safe to call from main thread — resets both sets
        self._mem        = []
        self._baseline   = 0
        self._cal        = False
        self._hand_was   = False
        self._hand_is    = False
        self._count_at   = 0
        self._confirm    = 0
        self._total      = 0
        self._log        = []
        with self._lock:
            self._d_total = 0
            self._d_log   = []
            self._d_cal   = False

    def recv(self, frame):
        img     = frame.to_ndarray(format="bgr24")
        results = model(img, conf=self.conf, imgsz=640, verbose=False)

        hands, lids = [], []
        for r in results:
            for box in r.boxes:
                coords = box.xyxy[0].tolist()
                label  = model.names[int(box.cls[0])]
                if label == 'hand':
                    hands.append(coords)
                else:
                    lids.append(coords)
                x1, y1, x2, y2 = map(int, coords)
                color = (160, 140, 100) if label == 'hand' else (48, 18, 196)
                cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
                cv2.putText(img, label, (x1, y1-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv_count  = len(lids)
        hand_contact = any(
            not (h[2]<l[0] or h[0]>l[2] or h[3]<l[1] or h[1]>l[3])
            for h in hands for l in lids
        )

        # Run logic — all on recv() thread, no lock needed here
        (self._mem, self._baseline, self._cal,
         self._hand_was, self._hand_is, self._count_at,
         self._confirm, self._total, self._log) = _apply_logic(
            cv_count, hand_contact,
            self._mem, self._baseline, self._cal,
            self._hand_was, self._hand_is, self._count_at,
            self._confirm, self._total, self._log
        )

        # Copy to display state under lock (very brief)
        with self._lock:
            self._d_total = self._total
            self._d_log   = list(self._log)
            self._d_cal   = self._cal

        # Overlay count on video
        overlay = f"INV: {self._total}" if self._cal else "Calibrating..."
        cv2.putText(img, overlay, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def get_display(self):
        with self._lock:
            return self._d_total, list(self._d_log), self._d_cal


# ── Shared logic state for demo/upload (plain dict) ───────
def make_state():
    return {
        "mem":       [],
        "baseline":  0,
        "cal":       False,
        "hand_was":  False,
        "hand_is":   False,
        "count_at":  0,
        "confirm":   0,
        "total_inv": 0,
        "log":       [],
    }

def run_logic_dict(cv_count, hand_contact, s):
    (s["mem"], s["baseline"], s["cal"],
     s["hand_was"], s["hand_is"], s["count_at"],
     s["confirm"], s["total_inv"], s["log"]) = _apply_logic(
        cv_count, hand_contact,
        s["mem"], s["baseline"], s["cal"],
        s["hand_was"], s["hand_is"], s["count_at"],
        s["confirm"], s["total_inv"], s["log"]
    )


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
                <span style='color:{BORDER}'>— awaiting events —</span>
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


# ── Video loop (demo + upload) ────────────────────────────
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

        # Inference every 3rd frame at 320px for speed
        if frame_count % 3 == 0:
            h, w   = frame.shape[:2]
            tw, th = 320, int(320 * h / w)
            small  = cv2.resize(frame, (tw, th))
            results = model(small, conf=conf, imgsz=tw, verbose=False)
            last_hands, last_lids = [], []
            sx, sy = w / tw, h / th
            for r in results:
                for box in r.boxes:
                    c      = box.xyxy[0].tolist()
                    scaled = [c[0]*sx, c[1]*sy, c[2]*sx, c[3]*sy]
                    label  = model.names[int(box.cls[0])]
                    if label == 'hand':
                        last_hands.append(scaled)
                    else:
                        last_lids.append(scaled)

        for c in last_hands:
            x1,y1,x2,y2 = map(int, c)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(160,140,100),2)
            cv2.putText(frame,'hand',(x1,y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(160,140,100),2)
        for c in last_lids:
            x1,y1,x2,y2 = map(int, c)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(48,18,196),2)
            cv2.putText(frame,'lid',(x1,y1-6),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(48,18,196),2)

        hand_contact = any(
            not (h[2]<l[0] or h[0]>l[2] or h[3]<l[1] or h[1]>l[3])
            for h in last_hands for l in last_lids
        )
        run_logic_dict(len(last_lids), hand_contact, s)
        frame_window.image(frame, channels="BGR", width='stretch')
        render_metrics(s["total_inv"], s["cal"])
        render_log(s["log"])
        time.sleep(0.03)


# ── Header ────────────────────────────────────────────────
hcol_logo, hcol_title, hcol_theme = st.columns([1, 7, 1])

with hcol_logo:
    if logo_b64:
        st.markdown(
            f"<img src='data:image/png;base64,{logo_b64}' "
            f"style='width:64px;margin-top:8px;'/>",
            unsafe_allow_html=True
        )

with hcol_title:
    st.markdown(f"""
        <div style='padding-top:10px'>
            <div style='font-size:21px;font-weight:700;color:{TEXT};letter-spacing:0.5px'>
                All-Clad Lid Inventory
            </div>
            <div style='font-size:11px;color:{TEXT_DIMMER};letter-spacing:2px;
                        text-transform:uppercase;margin-top:4px'>
                Computer Vision Tracking &nbsp;·&nbsp; Production Line 1
            </div>
        </div>
    """, unsafe_allow_html=True)

with hcol_theme:
    st.markdown("<div style='padding-top:12px'></div>", unsafe_allow_html=True)
    st.markdown("<div class='theme-btn'>", unsafe_allow_html=True)
    if st.button("☀ Light" if dark else "☾ Dark", key="theme_toggle"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    f"<div style='height:1px;"
    f"background:linear-gradient(90deg,transparent,{RED},transparent);"
    f"margin:14px 0 22px 0'></div>",
    unsafe_allow_html=True
)

# ── Sidebar ───────────────────────────────────────────────
st.sidebar.markdown(
    f"<p style='font-size:10px;font-weight:700;letter-spacing:2.5px;"
    f"text-transform:uppercase;color:{RED};border-bottom:1px solid {BORDER};"
    f"padding-bottom:8px;margin-bottom:16px;margin-top:8px'>System Control</p>",
    unsafe_allow_html=True
)
conf_threshold = st.sidebar.slider("Detection Confidence", 0.3, 0.9, 0.5)
reset_btn      = st.sidebar.button("⟳  Hard Reset")
mode           = st.sidebar.radio(
    "Input Mode",
    ["Live Camera (WebRTC)", "Demo Video", "Upload Video"]
)
st.sidebar.markdown(
    f"<div style='height:1px;background:{BORDER};margin:16px 0'></div>",
    unsafe_allow_html=True
)
st.sidebar.markdown(
    f"<p style='font-size:10px;color:{TEXT_DIMMER};letter-spacing:1px;"
    f"text-transform:uppercase;line-height:2.4'>"
    f"Model · lidDetection.pt<br>Buffer · 15 frames<br>Confirm · 8 frames</p>",
    unsafe_allow_html=True
)

# ── Main layout ───────────────────────────────────────────
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

    # ── Live Camera ──────────────────────────────────────
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
            total_inv, log, calibrated = ctx.video_processor.get_display()
            status = "ACTIVE" if ctx.state.playing else "IDLE"
            render_metrics(total_inv, calibrated, cam_status=status)
            render_log(log)
            time.sleep(0.5)
            st.rerun()
        else:
            render_metrics(0, False, cam_status="IDLE")
            render_log([])

    # ── Demo Video ───────────────────────────────────────
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
            st.session_state.demo_cap     = cv2.VideoCapture("demo_video.mp4")
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
            process_video_loop(
                st.session_state.demo_cap, frame_window, s_demo, conf_threshold
            )
        else:
            render_metrics(s_demo["total_inv"], s_demo["cal"])
            render_log(s_demo["log"])
            st.markdown(
                "<div class='placeholder-box'>Press ▶ Start Demo to begin</div>",
                unsafe_allow_html=True
            )

    # ── Upload Video ─────────────────────────────────────
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
                st.session_state.upload_cap     = cv2.VideoCapture(tfile.name)
                st.session_state.upload_path    = tfile.name
                st.session_state.upload_name    = uploaded.name
                st.session_state.upload_running = False
                st.session_state.s_upload       = make_state()
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
                    st.session_state.upload_cap, frame_window_up, s_upload, conf_threshold
                )
            else:
                render_metrics(s_upload["total_inv"], s_upload["cal"])
                render_log(s_upload["log"])
                st.markdown(
                    "<div class='placeholder-box'>Press ▶ Start to begin</div>",
                    unsafe_allow_html=True
                )
        else:
            render_metrics(0, False)
            render_log([])
            st.markdown(
                "<div class='placeholder-box'>Upload a .mp4 / .mov / .avi to begin</div>",
                unsafe_allow_html=True
            )


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
