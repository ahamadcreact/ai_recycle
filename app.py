import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Nav Jeevan Mission School | Recycle Checker",
                   page_icon="♻️", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Pacifico&family=Poppins:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
.stApp { background: linear-gradient(160deg,#fff4dc 0%,#e3fbec 50%,#e1f0ff 100%); }
.stApp p, .stApp label, .stApp span, .stApp li { color:#1b2a41; }
header[data-testid="stHeader"] { background: transparent; }

.hero {
  background: linear-gradient(120deg,#ff6a3d,#ff3d81,#7b2ff7,#1fa2ff);
  background-size: 300% 300%;
  animation: flow 8s ease infinite;
  border-radius: 28px; padding: 30px 20px; text-align:center;
  box-shadow: 0 12px 30px rgba(123,47,247,.35); margin-bottom: 22px;
}
@keyframes flow { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
.hero .school { font-family:'Pacifico',cursive; font-size:2.6rem; color:#fff !important;
  text-shadow: 3px 3px 0 rgba(0,0,0,.25); line-height:1.2; }
.hero .tag { color:#fff !important; font-weight:600; letter-spacing:1px; margin-top:6px; font-size:1.05rem; }
.hero .pill { display:inline-block; margin-top:14px; background:rgba(255,255,255,.95);
  color:#7b2ff7 !important; font-weight:700; padding:6px 18px; border-radius:999px; }

.card { border-radius:22px; padding:22px; text-align:center; margin-top:16px;
  box-shadow:0 10px 24px rgba(0,0,0,.15); }
.card .emoji { font-size:3.5rem; }
.card .title { font-size:1.7rem; font-weight:700; margin:4px 0; color:#fff !important; }
.card .tip { color:#fff !important; font-size:1rem; }
.card .conf { display:inline-block; margin-top:10px; background:rgba(255,255,255,.25);
  color:#fff !important; padding:4px 16px; border-radius:999px; font-weight:600; }

.bar-row { display:flex; align-items:center; gap:10px; margin:6px 0; }
.bar-name { width:110px; font-weight:600; color:#1b2a41 !important; }
.bar-bg { flex:1; background:#e4e9f2; border-radius:999px; height:14px; overflow:hidden; }
.bar-fill { height:100%; border-radius:999px; }
.bar-val { width:48px; text-align:right; font-weight:600; color:#1b2a41 !important; }

.footer { text-align:center; margin-top:34px; padding:14px; border-radius:16px;
  background:rgba(255,255,255,.7); font-weight:600; color:#1b2a41 !important; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg,#1fa2ff22,#7b2ff722); }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load():
    model = tf.keras.models.load_model("recycle_model.keras")
    names = json.load(open("class_names.json"))
    return model, names


model, names = load()

INFO = {
    "cardboard": ("📦", "#f59e0b", True, "Flatten it, keep it dry, and place it in the recycling bin."),
    "glass":     ("🍾", "#10b981", True, "Rinse and dry it. Handle broken glass with care."),
    "metal":     ("🥫", "#3b82f6", True, "Rinse and crush it, then place it in the recycling bin."),
    "paper":     ("📰", "#8b5cf6", True, "Only clean, dry paper can be recycled."),
    "plastic":   ("🧴", "#06b6d4", True, "Rinse it and remove the cap. Plastic bags are usually not recyclable."),
    "trash":     ("🗑️", "#ef4444", False, "This goes into the regular waste bin."),
}

# ---------- Header ----------
st.markdown("""
<div class="hero">
  <div class="school">Nav Jeevan Mission School</div>
  <div class="tag">🌍 Clean Earth · Green Future 🌱</div>
  <div class="pill">♻️ Smart Recycle Detector (AI)</div>
</div>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### 🏫 Nav Jeevan Mission School")
    st.markdown("**AI Project: Is it recyclable?**")
    st.markdown("""
**How to use**
1. Take a photo with the camera or upload one
2. The AI recognises the object
3. See the result and a disposal tip
""")
    st.markdown("**It can recognise these 6 items**")
    for n in names:
        st.markdown(f"{INFO[n][0]} {n.title()}")
    st.caption("Model: MobileNetV2 (Transfer Learning) · Dataset: TrashNet")

# ---------- Input ----------
tab1, tab2 = st.tabs(["📷 Use camera", "🖼️ Upload photo"])
with tab1:
    cam = st.camera_input("Hold the object in front of the camera")
with tab2:
    up = st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png"])

img_file = cam or up

if not img_file:
    st.info("👆 Use the camera or upload a photo above. The result will appear here.")
else:
    img = Image.open(img_file).convert("RGB")
    st.image(img, use_container_width=True)

    x = np.expand_dims(np.array(img.resize((224, 224)), dtype="float32"), 0)
    probs = model.predict(x, verbose=0)[0]
    i = int(np.argmax(probs))
    label, conf = names[i], float(probs[i])
    emoji, color, recyclable, tip = INFO[label]

    if conf < 0.5:
        st.markdown(f"""
        <div class="card" style="background:linear-gradient(135deg,#f59e0b,#fbbf24);">
          <div class="emoji">🤔</div>
          <div class="title">Not sure</div>
          <div class="tip">The AI is only {conf:.0%} sure. Try better lighting or a different angle.</div>
        </div>""", unsafe_allow_html=True)
    else:
        verdict = "Recyclable ✅" if recyclable else "Not recyclable ❌"
        bg = "linear-gradient(135deg,#059669,#34d399)" if recyclable else "linear-gradient(135deg,#dc2626,#f87171)"
        st.markdown(f"""
        <div class="card" style="background:{bg};">
          <div class="emoji">{emoji}</div>
          <div class="title">{label.title()}: {verdict}</div>
          <div class="tip">{tip}</div>
          <div class="conf">Confidence: {conf:.0%}</div>
        </div>""", unsafe_allow_html=True)
        if recyclable and conf >= 0.7:
            st.balloons()

    st.markdown("#### 📊 What does the AI think?")
    bars = ""
    for n, p in sorted(zip(names, probs), key=lambda t: -t[1]):
        c = INFO[n][1]
        bars += f"""<div class="bar-row"><div class="bar-name">{INFO[n][0]} {n.title()}</div>
        <div class="bar-bg"><div class="bar-fill" style="width:{p*100:.0f}%;background:{c};"></div></div>
        <div class="bar-val">{p:.0%}</div></div>"""
    st.markdown(bars, unsafe_allow_html=True)

st.markdown("""
<div class="footer">🌿 Created for Nav Jeevan Mission School · Let's keep our Earth green 🌿</div>
""", unsafe_allow_html=True)