import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import time
import datetime
import io
import base64

st.set_page_config(
    page_title="VisionAI Pro | Object Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: #F0F2F5 !important;
    color: #0F172A !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid #E2E8F0 !important;
}
[data-testid="stSidebar"] * { color: #0F172A !important; font-family: 'Inter', sans-serif !important; }
.block-container { padding: 1.2rem 1.8rem !important; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── TOP NAV BAR ── */
.topbar {
    background: #0F172A;
    border-radius: 12px;
    padding: 0.8rem 1.6rem;
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 1.2rem;
}
.topbar-left { display: flex; align-items: center; gap: 14px; }
.topbar-logo {
    width: 34px; height: 34px; border-radius: 8px;
    background: linear-gradient(135deg, #2563EB, #1D4ED8);
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
}
.topbar-title { font-size: 16px; font-weight: 700; color: #FFFFFF; letter-spacing: -0.3px; }
.topbar-sub { font-size: 11px; color: #64748B; font-family: 'JetBrains Mono', monospace; }
.topbar-right { display: flex; align-items: center; gap: 12px; }
.tb-badge {
    background: #1E293B; border: 1px solid #334155;
    border-radius: 6px; padding: 4px 12px;
    font-size: 11px; font-family: 'JetBrains Mono', monospace; color: #94A3B8;
}
.tb-status {
    display: flex; align-items: center; gap: 6px;
    background: #052E16; border: 1px solid #166534;
    border-radius: 6px; padding: 4px 12px;
    font-size: 11px; font-family: 'JetBrains Mono', monospace; color: #4ADE80;
}
.tb-dot { width: 6px; height: 6px; border-radius: 50%; background: #22C55E; animation: blink 1.5s infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }

/* ── KPI CARDS ── */
.kpi-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 10px; margin-bottom: 1.2rem; }
.kpi {
    background: #FFFFFF; border: 1px solid #E2E8F0;
    border-radius: 10px; padding: 1rem 1.2rem;
    box-shadow: 0 1px 3px rgba(15,23,42,0.06);
    display: flex; justify-content: space-between; align-items: flex-start;
}
.kpi-left {}
.kpi-label { font-size: 11px; font-weight: 500; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 6px; }
.kpi-value { font-size: 28px; font-weight: 700; color: #0F172A; line-height: 1; }
.kpi-sub { font-size: 11px; color: #CBD5E1; margin-top: 3px; }
.kpi-icon {
    width: 36px; height: 36px; border-radius: 8px;
    display: flex; align-items: center; justify-content: center; font-size: 16px;
}
.kpi-icon.blue { background: #EFF6FF; }
.kpi-icon.green { background: #F0FDF4; }
.kpi-icon.amber { background: #FFFBEB; }
.kpi-icon.violet { background: #F5F3FF; }

/* ── SECTION HEADER ── */
.sec-hdr {
    font-size: 11px; font-weight: 600; color: #94A3B8;
    text-transform: uppercase; letter-spacing: 1.2px;
    display: flex; align-items: center; gap: 8px; margin-bottom: 10px;
}
.sec-hdr-line { flex:1; height:1px; background:#E2E8F0; }

/* ── CARD ── */
.card {
    background: #FFFFFF; border: 1px solid #E2E8F0;
    border-radius: 12px; padding: 1.2rem;
    box-shadow: 0 1px 3px rgba(15,23,42,0.05);
}

/* ── DETECTION TAGS ── */
.tag-grid { display: flex; flex-wrap: wrap; gap: 6px; }
.dtag {
    display: flex; align-items: center; gap: 5px;
    background: #F8FAFC; border: 1px solid #E2E8F0;
    border-radius: 6px; padding: 4px 10px;
    font-size: 12px; font-family: 'JetBrains Mono', monospace; color: #334155;
}
.dtag-conf {
    background: #EFF6FF; color: #2563EB;
    border-radius: 4px; padding: 0px 5px; font-size: 11px;
}

/* ── TABLE ── */
.det-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.det-table th {
    background: #F8FAFC; color: #64748B;
    font-size: 10px; text-transform: uppercase; letter-spacing: 0.8px;
    padding: 8px 12px; text-align: left; border-bottom: 1px solid #E2E8F0;
    font-weight: 600;
}
.det-table td { padding: 8px 12px; border-bottom: 1px solid #F1F5F9; color: #334155; }
.det-table tr:last-child td { border-bottom: none; }
.det-table tr:hover td { background: #F8FAFC; }
.rank-badge {
    display: inline-flex; align-items: center; justify-content: center;
    width: 20px; height: 20px; border-radius: 50%;
    background: #EFF6FF; color: #2563EB;
    font-size: 10px; font-weight: 700;
}

/* ── BAR CHART ── */
.bar-row { margin-bottom: 9px; }
.bar-meta { display: flex; justify-content: space-between; font-size: 12px; color: #475569; margin-bottom: 3px; }
.bar-bg { height: 6px; background: #F1F5F9; border-radius: 99px; }
.bar-fill { height: 6px; border-radius: 99px; }

/* ── PIE CHART (CSS) ── */
.pie-wrap { display: flex; align-items: center; gap: 20px; }
.legend-item { display: flex; align-items: center; gap: 7px; font-size: 12px; color: #475569; margin-bottom: 6px; }
.legend-dot { width: 10px; height: 10px; border-radius: 3px; flex-shrink: 0; }

/* ── LOG ── */
.log-row {
    display: flex; gap: 10px; align-items: flex-start;
    padding: 6px 0; border-bottom: 1px solid #F1F5F9;
    font-size: 12px;
}
.log-t { color: #CBD5E1; font-family: 'JetBrains Mono', monospace; min-width: 62px; }
.log-o { color: #2563EB; font-weight: 500; }
.log-c { color: #94A3B8; }

/* ── EMPTY STATE ── */
.empty {
    height: 280px; display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    border: 1.5px dashed #E2E8F0; border-radius: 10px;
    color: #CBD5E1; font-size: 12px; gap: 8px;
    font-family: 'JetBrains Mono', monospace;
}

/* ── STREAMLIT ── */
.stButton > button {
    background: #2563EB !important; color: #fff !important;
    border: none !important; border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important; font-weight: 600 !important;
    font-size: 13px !important; padding: 0.45rem 1.2rem !important;
    box-shadow: 0 1px 4px rgba(37,99,235,0.3) !important;
}
.stButton > button:hover { background: #1D4ED8 !important; }
.stSlider > div > div > div { background: #2563EB !important; }
div[data-testid="stImage"] img { border-radius: 10px; border: 1px solid #E2E8F0; }
.stCheckbox label { font-size: 13px !important; font-family: 'Inter', sans-serif !important; color: #334155 !important; }
.stFileUploader { background: #FFFFFF !important; border-radius: 10px !important; }
.stRadio label { font-size: 12px !important; font-family: 'Inter', sans-serif !important; }
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ────────────────────────────────────────────────────────────
for k, v in [("total", 0), ("frames", 0), ("log", []), ("counts", {}), ("t0", time.time())]:
    if k not in st.session_state:
        st.session_state[k] = v

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")
model = load_model()

# ── TOPBAR ───────────────────────────────────────────────────────────────────
e = int(time.time() - st.session_state.t0)
h, m = divmod(e // 60, 60); s = e % 60
st.markdown(f"""
<div class="topbar">
  <div class="topbar-left">
    <div class="topbar-logo">🎯</div>
    <div>
      <div class="topbar-title">VisionAI Pro</div>
      <div class="topbar-sub">YOLOv8 Object Detection Platform</div>
    </div>
  </div>
  <div class="topbar-right">
    <div class="tb-badge">MODEL · YOLOv8n</div>
    <div class="tb-badge">CLASSES · 80</div>
    <div class="tb-badge">SESSION · {h:02d}:{m:02d}:{s:02d}</div>
    <div class="tb-status"><div class="tb-dot"></div>ONLINE</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Controls")
    st.markdown("---")
    mode = st.radio("Input Mode", ["📁 Image Upload", "📷 Webcam"], label_visibility="collapsed")
    st.markdown("**Confidence Threshold**")
    confidence = st.slider("conf", 0.1, 1.0, 0.25, 0.05, label_visibility="collapsed")
    st.markdown(f"<p style='font-size:11px;color:#94A3B8;font-family:JetBrains Mono,monospace;margin-top:-8px'>Min score: {confidence:.0%}</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**System Info**")
    st.markdown(f"""
    <div style='font-size:11px;font-family:JetBrains Mono,monospace;color:#94A3B8;line-height:2.2'>
    Engine &nbsp;&nbsp;<span style='color:#2563EB'>PyTorch</span><br>
    Model &nbsp;&nbsp;&nbsp;<span style='color:#2563EB'>YOLOv8 Nano</span><br>
    Dataset &nbsp;<span style='color:#2563EB'>COCO 80cls</span><br>
    Status &nbsp;&nbsp;<span style='color:#22C55E'>● Active</span>
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    if st.button("↺ Reset Dashboard"):
        for k, v in [("total",0),("frames",0),("log",[]),("counts",{}),("t0",time.time())]:
            st.session_state[k] = v
        st.rerun()

    st.markdown("---")
    st.markdown("**80 Detectable Classes**")
    st.markdown("<p style='font-size:11px;color:#94A3B8;margin-bottom:10px'>COCO dataset · 11 categories</p>", unsafe_allow_html=True)

    coco_html = """
    <style>
    .cat-block { margin-bottom: 12px; }
    .cat-title {
        font-size: 10px; font-weight: 700; color: #64748B;
        text-transform: uppercase; letter-spacing: 1px;
        font-family: JetBrains Mono, monospace;
        margin-bottom: 5px; padding-bottom: 3px;
        border-bottom: 1px solid #F1F5F9;
    }
    .cat-tags { display: flex; flex-wrap: wrap; gap: 4px; }
    .ctag {
        background: #EFF6FF; color: #2563EB;
        border: 1px solid #BFDBFE; border-radius: 4px;
        padding: 2px 7px; font-size: 10px;
        font-family: JetBrains Mono, monospace;
    }
    </style>
    <div class="cat-block"><div class="cat-title">People</div><div class="cat-tags"><span class="ctag">person</span></div></div>
    <div class="cat-block"><div class="cat-title">Vehicles</div><div class="cat-tags"><span class="ctag">bicycle</span><span class="ctag">car</span><span class="ctag">motorcycle</span><span class="ctag">airplane</span><span class="ctag">bus</span><span class="ctag">train</span><span class="ctag">truck</span><span class="ctag">boat</span></div></div>
    <div class="cat-block"><div class="cat-title">Outdoor</div><div class="cat-tags"><span class="ctag">traffic light</span><span class="ctag">fire hydrant</span><span class="ctag">stop sign</span><span class="ctag">parking meter</span><span class="ctag">bench</span></div></div>
    <div class="cat-block"><div class="cat-title">Animals</div><div class="cat-tags"><span class="ctag">bird</span><span class="ctag">cat</span><span class="ctag">dog</span><span class="ctag">horse</span><span class="ctag">sheep</span><span class="ctag">cow</span><span class="ctag">elephant</span><span class="ctag">bear</span><span class="ctag">zebra</span><span class="ctag">giraffe</span></div></div>
    <div class="cat-block"><div class="cat-title">Accessories</div><div class="cat-tags"><span class="ctag">backpack</span><span class="ctag">umbrella</span><span class="ctag">handbag</span><span class="ctag">tie</span><span class="ctag">suitcase</span></div></div>
    <div class="cat-block"><div class="cat-title">Sports</div><div class="cat-tags"><span class="ctag">frisbee</span><span class="ctag">skis</span><span class="ctag">snowboard</span><span class="ctag">sports ball</span><span class="ctag">kite</span><span class="ctag">baseball bat</span><span class="ctag">baseball glove</span><span class="ctag">skateboard</span><span class="ctag">surfboard</span><span class="ctag">tennis racket</span></div></div>
    <div class="cat-block"><div class="cat-title">Kitchen</div><div class="cat-tags"><span class="ctag">bottle</span><span class="ctag">wine glass</span><span class="ctag">cup</span><span class="ctag">fork</span><span class="ctag">knife</span><span class="ctag">spoon</span><span class="ctag">bowl</span></div></div>
    <div class="cat-block"><div class="cat-title">Food</div><div class="cat-tags"><span class="ctag">banana</span><span class="ctag">apple</span><span class="ctag">sandwich</span><span class="ctag">orange</span><span class="ctag">broccoli</span><span class="ctag">carrot</span><span class="ctag">hot dog</span><span class="ctag">pizza</span><span class="ctag">donut</span><span class="ctag">cake</span></div></div>
    <div class="cat-block"><div class="cat-title">Furniture</div><div class="cat-tags"><span class="ctag">chair</span><span class="ctag">couch</span><span class="ctag">potted plant</span><span class="ctag">bed</span><span class="ctag">dining table</span><span class="ctag">toilet</span></div></div>
    <div class="cat-block"><div class="cat-title">Electronics</div><div class="cat-tags"><span class="ctag">tv</span><span class="ctag">laptop</span><span class="ctag">mouse</span><span class="ctag">remote</span><span class="ctag">keyboard</span><span class="ctag">cell phone</span><span class="ctag">microwave</span><span class="ctag">oven</span><span class="ctag">toaster</span><span class="ctag">sink</span><span class="ctag">refrigerator</span></div></div>
    <div class="cat-block"><div class="cat-title">Misc</div><div class="cat-tags"><span class="ctag">book</span><span class="ctag">clock</span><span class="ctag">vase</span><span class="ctag">scissors</span><span class="ctag">teddy bear</span><span class="ctag">hair drier</span><span class="ctag">toothbrush</span></div></div>
    """
    st.markdown(coco_html, unsafe_allow_html=True)

# ── KPI ROW ──────────────────────────────────────────────────────────────────
top = max(st.session_state.counts, key=st.session_state.counts.get) if st.session_state.counts else "—"
uniq = len(st.session_state.counts)
st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi">
    <div class="kpi-left">
      <div class="kpi-label">Total Detections</div>
      <div class="kpi-value">{st.session_state.total}</div>
      <div class="kpi-sub">objects found</div>
    </div>
    <div class="kpi-icon blue">🔍</div>
  </div>
  <div class="kpi">
    <div class="kpi-left">
      <div class="kpi-label">Frames Processed</div>
      <div class="kpi-value">{st.session_state.frames}</div>
      <div class="kpi-sub">inference runs</div>
    </div>
    <div class="kpi-icon green">🖼️</div>
  </div>
  <div class="kpi">
    <div class="kpi-left">
      <div class="kpi-label">Unique Classes</div>
      <div class="kpi-value">{uniq}</div>
      <div class="kpi-sub">distinct types</div>
    </div>
    <div class="kpi-icon amber">📦</div>
  </div>
  <div class="kpi">
    <div class="kpi-left">
      <div class="kpi-label">Top Object</div>
      <div class="kpi-value" style="font-size:{'16px' if len(top)>6 else '22px'}">{top.capitalize()}</div>
      <div class="kpi-sub">most frequent</div>
    </div>
    <div class="kpi-icon violet">🏆</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── DETECT FUNCTION ──────────────────────────────────────────────────────────
def detect(image, conf):
    arr = np.array(image)
    res = model(arr, conf=conf, verbose=False)
    # Draw big bold labels manually on top of YOLOv8 annotations
    annotated = cv2.cvtColor(res[0].plot(), cv2.COLOR_BGR2RGB)
    dets = []
    for box in res[0].boxes:
        lbl = model.names[int(box.cls[0])]
        c = float(box.conf[0])
        dets.append({"label": lbl.capitalize(), "conf": c})
        st.session_state.counts[lbl] = st.session_state.counts.get(lbl, 0) + 1
        st.session_state.total += 1
        st.session_state.log.insert(0, {
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "obj": lbl.capitalize(), "conf": f"{c*100:.0f}%"
        })
        # Draw big visible label on image
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        label_text = f"{lbl.upper()}  {c*100:.0f}%"
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = max(0.7, min(1.4, (x2-x1)/200))
        thickness = 2
        (tw, th), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
        # Background rectangle for label
        cv2.rectangle(annotated, (x1, max(0,y1-th-16)), (x1+tw+12, y1), (37,99,235), -1)
        # Label text
        cv2.putText(annotated, label_text, (x1+6, max(th, y1-6)), font, font_scale, (255,255,255), thickness, cv2.LINE_AA)
    st.session_state.log = st.session_state.log[:40]
    st.session_state.frames += 1
    return annotated, dets

def img_to_bytes(img_array):
    pil = Image.fromarray(img_array)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()

# ── LAYOUT ───────────────────────────────────────────────────────────────────
c1, c2 = st.columns([3, 2], gap="medium")
current_dets = []
current_img = None

with c1:
    st.markdown('<div class="sec-hdr">Detection Feed <div class="sec-hdr-line"></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)

    if mode == "📁 Image Upload":
        uploaded = st.file_uploader("", type=["jpg","jpeg","png"], label_visibility="collapsed")
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            with st.spinner("Running inference..."):
                annotated, current_dets = detect(image, confidence)
            current_img = annotated

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("<p style='font-size:11px;color:#94A3B8;font-weight:600;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:6px'>Original</p>", unsafe_allow_html=True)
                st.image(image, use_container_width=True)
            with col_b:
                st.markdown("<p style='font-size:11px;color:#94A3B8;font-weight:600;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:6px'>Detected</p>", unsafe_allow_html=True)
                st.image(annotated, use_container_width=True)

            # Download button
            st.markdown("<div style='margin-top:10px'>", unsafe_allow_html=True)
            st.download_button(
                label="⬇ Download Detected Image",
                data=img_to_bytes(annotated),
                file_name=f"detection_{datetime.datetime.now().strftime('%H%M%S')}.png",
                mime="image/png"
            )
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown('<div class="empty"><div style="font-size:32px">📁</div><div>Upload an image to begin</div><div style="font-size:10px;color:#E2E8F0">JPG · JPEG · PNG</div></div>', unsafe_allow_html=True)

    else:
        run = st.checkbox("▶ Start Webcam", value=False)
        ph = st.empty()
        if run:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Cannot open webcam.")
            else:
                stop = st.button("⏹ Stop")
                while not stop:
                    ret, frame = cap.read()
                    if not ret: break
                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    annotated, current_dets = detect(img, confidence)
                    current_img = annotated
                    ph.image(annotated, channels="RGB", use_container_width=True)
                cap.release()
        else:
            ph.markdown('<div class="empty"><div style="font-size:32px">📷</div><div>Enable webcam to start</div></div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── DETECTION TABLE ──
    if current_dets or st.session_state.counts:
        st.markdown('<div style="margin-top:12px"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-hdr">Object Count Table <div class="sec-hdr-line"></div></div>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)

        if st.session_state.counts:
            sorted_counts = sorted(st.session_state.counts.items(), key=lambda x: x[1], reverse=True)
            rows = "".join([
                f"""<tr>
                  <td><span class="rank-badge">{i+1}</span></td>
                  <td><strong>{obj.capitalize()}</strong></td>
                  <td>{cnt}</td>
                  <td>
                    <div style="display:flex;align-items:center;gap:8px">
                      <div style="flex:1;height:5px;background:#F1F5F9;border-radius:99px">
                        <div style="width:{int(cnt/sorted_counts[0][1]*100)}%;height:5px;background:#2563EB;border-radius:99px"></div>
                      </div>
                      <span style="font-size:11px;color:#94A3B8">{int(cnt/sum(st.session_state.counts.values())*100)}%</span>
                    </div>
                  </td>
                </tr>"""
                for i, (obj, cnt) in enumerate(sorted_counts)
            ])
            st.markdown(f"""
            <table class="det-table">
              <thead><tr>
                <th>#</th><th>Object</th><th>Count</th><th>Share</th>
              </tr></thead>
              <tbody>{rows}</tbody>
            </table>""", unsafe_allow_html=True)
        else:
            st.markdown('<p style="font-size:12px;color:#CBD5E1;font-family:JetBrains Mono,monospace">No data yet</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

with c2:
    # ── LIVE DETECTIONS ──
    st.markdown('<div class="sec-hdr">Live Detections <div class="sec-hdr-line"></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="card" style="margin-bottom:10px">', unsafe_allow_html=True)
    if current_dets:
        tags = '<div class="tag-grid">'
        for d in current_dets:
            tags += f'<div class="dtag">▸ {d["label"]} <span class="dtag-conf">{d["conf"]*100:.0f}%</span></div>'
        tags += '</div>'
        st.markdown(tags, unsafe_allow_html=True)
    else:
        st.markdown('<p style="font-size:12px;color:#CBD5E1;font-family:JetBrains Mono,monospace">Awaiting detection...</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── BAR CHART ──
    st.markdown('<div class="sec-hdr">Frequency Chart <div class="sec-hdr-line"></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="card" style="margin-bottom:10px">', unsafe_allow_html=True)
    colors = ["#2563EB","#7C3AED","#0891B2","#059669","#D97706","#DC2626"]
    if st.session_state.counts:
        top6 = sorted(st.session_state.counts.items(), key=lambda x:x[1], reverse=True)[:6]
        mx = top6[0][1]
        bars = ""
        for i,(obj,cnt) in enumerate(top6):
            pct = int(cnt/mx*100)
            bars += f"""<div class="bar-row">
              <div class="bar-meta"><span>{obj.capitalize()}</span><span style="font-family:JetBrains Mono,monospace">{cnt}</span></div>
              <div class="bar-bg"><div class="bar-fill" style="width:{pct}%;background:{colors[i%len(colors)]}"></div></div>
            </div>"""
        st.markdown(bars, unsafe_allow_html=True)
    else:
        st.markdown('<p style="font-size:12px;color:#CBD5E1;font-family:JetBrains Mono,monospace">No data yet</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── PIE / DISTRIBUTION ──
    st.markdown('<div class="sec-hdr">Distribution <div class="sec-hdr-line"></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="card" style="margin-bottom:10px">', unsafe_allow_html=True)
    if st.session_state.counts and sum(st.session_state.counts.values()) > 0:
        total_c = sum(st.session_state.counts.values())
        top5 = sorted(st.session_state.counts.items(), key=lambda x:x[1], reverse=True)[:5]
        legend = ""
        for i,(obj,cnt) in enumerate(top5):
            pct = int(cnt/total_c*100)
            legend += f"""<div class="legend-item">
              <div class="legend-dot" style="background:{colors[i%len(colors)]}"></div>
              <span style="flex:1">{obj.capitalize()}</span>
              <span style="font-family:JetBrains Mono,monospace;color:#94A3B8">{pct}%</span>
            </div>"""
        st.markdown(f'<div class="pie-wrap"><div style="flex:1">{legend}</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<p style="font-size:12px;color:#CBD5E1;font-family:JetBrains Mono,monospace">No data yet</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── LOG ──
    st.markdown('<div class="sec-hdr">Detection Log <div class="sec-hdr-line"></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    if st.session_state.log:
        log_html = ""
        for e in st.session_state.log[:10]:
            log_html += f"""<div class="log-row">
              <span class="log-t">{e['time']}</span>
              <span style="flex:1;color:#475569">Found <span class="log-o">{e['obj']}</span></span>
              <span class="log-c">{e['conf']}</span>
            </div>"""
        st.markdown(log_html, unsafe_allow_html=True)
    else:
        st.markdown('<p style="font-size:12px;color:#CBD5E1;font-family:JetBrains Mono,monospace">No activity yet</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)