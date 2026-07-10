import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
import os
import time
import google.generativeai as genai
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image

# --- 🔧 PROXY SETUP (China AntLink Fix: 7892) ---
os.environ["http_proxy"] = "http://127.0.0.1:7892"
os.environ["https_proxy"] = "http://127.0.0.1:7892"

# --- 🔐 API KEY ---
GOOGLE_API_KEY = "AIzaSyCwupoE0PFBkQ84kAy43F9fzS5FAIh9E60"

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Advanced AI Caption Generator",
    page_icon="🌈",
    layout="wide"
)

# --- 🎨 YOUR CUSTOM CSS (Rainbow Sidebar & Clean UI) ---
st.markdown("""
<style>
    /* 1. RAINBOW SIDEBAR BACKGROUND (ANIMATED) */
    [data-testid="stSidebar"] {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }

    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Sidebar Text Color */
    [data-testid="stSidebar"] * {
        color: white !important;
    }

    /* 2. MAIN TITLE RAINBOW */
    .rainbow-text {
        background-image: linear-gradient(to left, violet, indigo, blue, green, yellow, orange, red);
        -webkit-background-clip: text;
        color: transparent;
        font-weight: 800;
        font-size: 48px;
        text-align: center;
        margin-bottom: 10px;
    }

    .dev-credit {
        text-align: center;
        color: #666;
        font-size: 16px;
        font-weight: bold;
        margin-bottom: 40px;
        font-family: monospace;
    }

    /* 3. RESULT CARD DESIGN (CLEAN WHITE) */
    .result-card {
        background-color: #ffffff;
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        text-align: center;
        border: 2px solid #f0f0f0;
        animation: slideUp 0.5s ease-out;
    }

    .caption-text {
        color: #333;
        font-size: 26px;
        font-weight: 700;
        font-family: 'Segoe UI', sans-serif;
        margin-top: 15px;
    }

    .badge {
        background-color: #e3f2fd;
        color: #1565c0;
        padding: 5px 15px;
        border-radius: 50px;
        font-size: 12px;
        font-weight: bold;
        text-transform: uppercase;
    }

    @keyframes slideUp {
        from { transform: translateY(20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }

    /* Upload Box */
    .stFileUploader {
        background: #f9f9f9;
        border: 2px dashed #ddd;
        border-radius: 15px;
        padding: 20px;
    }

    /* Button */
    .stButton>button {
        background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
        padding: 10px 0;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')


# --- LOADING FUNCTIONS ---
@st.cache_resource
def load_resources():
    try:
        with open(os.path.join(MODELS_DIR, 'tokenizer.pkl'), 'rb') as f:
            return pickle.load(f)
    except:
        return None


@st.cache_resource
def load_local_model(model_name):
    from tensorflow.keras.applications.xception import Xception, preprocess_input as xception_pp
    from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input as ir_pp
    from tensorflow.keras.applications.densenet import DenseNet201, preprocess_input as dn_pp

    try:
        if "Xception" in model_name:
            path = os.path.join(MODELS_DIR, 'model_xception.h5')
            base = Xception(weights='imagenet')
            base = Model(inputs=base.inputs, outputs=base.layers[-2].output)
            pp = xception_pp
            size = (299, 299)
        elif "Inception" in model_name:
            path = os.path.join(MODELS_DIR, 'model_inception_resnet_v2.h5')
            base = InceptionResNetV2(weights='imagenet')
            base = Model(inputs=base.inputs, outputs=base.layers[-2].output)
            pp = ir_pp
            size = (299, 299)
        else:
            path = os.path.join(MODELS_DIR, 'model_densenet201.h5')
            base = DenseNet201(weights='imagenet')
            base = Model(inputs=base.inputs, outputs=base.layers[-2].output)
            pp = dn_pp
            size = (224, 224)

        loaded_model = load_model(path)
        return base, loaded_model, pp, size
    except:
        return None, None, None, None


# --- GENERATION LOGIC ---
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer: return word
    return None


def greedy_search(model, feature, tokenizer, max_length=34):
    in_text = 'startseq'
    for i in range(max_length):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_length)[0].reshape(1, max_length)
        yhat = model.predict([feature, seq], verbose=0)
        word = idx_to_word(np.argmax(yhat), tokenizer)
        if word is None: break
        in_text += ' ' + word
        if word == 'endseq': break
    return in_text.replace('startseq', '').replace('endseq', '')


def beam_search(model, feature, tokenizer, max_length=34, k=5):
    start = [tokenizer.word_index['startseq']]
    sequences = [[start, 0.0]]
    while len(sequences[0][0]) < max_length:
        temp = []
        for seq, score in sequences:
            if seq[-1] == tokenizer.word_index.get('endseq'):
                temp.append([seq, score])
                continue
            padded_seq = pad_sequences([seq], maxlen=max_length)[0].reshape(1, max_length)
            preds = model.predict([feature, padded_seq], verbose=0)
            top_words = np.argsort(preds[0])[-k:]
            for word in top_words:
                next_seq, new_score = seq + [word], score - np.log(preds[0][word])
                temp.append([next_seq, new_score])
        sequences = temp
        sequences = sorted(sequences, key=lambda l: l[1])[:k]
        if all(seq[0][-1] == tokenizer.word_index.get('endseq') for seq in sequences): break

    best_seq = sequences[0][0]
    final_caption = [idx_to_word(i, tokenizer) for i in best_seq]
    return " ".join(final_caption).replace('startseq', '').replace('endseq', '')


def generate_gemini(image):
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(
        ["Describe this image accurately in one sentence.", image],
        request_options={"timeout": 60}
    )
    return response.text


# --- APP HEADER ---
st.markdown('<div class="rainbow-text">🌈 Advanced AI Caption Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="dev-credit">Developer: Golam Rohman Ruhit</div>', unsafe_allow_html=True)
st.markdown("---")

# --- SIDEBAR ---
st.sidebar.markdown("## ⚙️ **Control Panel**")
engine = st.sidebar.radio(
    "Choose Engine:",
    ["Google Gemini (100% Accurate)", "My Custom AI (Local)"]
)

if engine == "My Custom AI (Local)":
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧠 AI Architecture")
    model_choice = st.sidebar.selectbox("Select Model:", ["Xception", "InceptionResNetV2", "DenseNet201"])

    st.sidebar.markdown("### 🔍 Search Strategy")
    search_mode = st.sidebar.radio("Decoding Method:", ["Beam Search (Accurate)", "Greedy Search (Fast)"])

    if "Beam" in search_mode:
        beam_k = st.sidebar.slider("Beam Size (k):", 3, 10, 5)
else:
    st.sidebar.success("✨ Gemini 1.5 Flash Active")

# --- MAIN UI ---
col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.info("📤 **Upload Image**")
    uploaded_file = st.file_uploader("", type=['jpg', 'png', 'jpeg'])
    if uploaded_file:
        image = Image.open(uploaded_file)
        # Clean Shadow Container
        st.markdown('<div style="box-shadow: 0 5px 15px rgba(0,0,0,0.1); border-radius: 10px; overflow: hidden;">',
                    unsafe_allow_html=True)
        st.image(image, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.success(f"🤖 **System Ready: {engine.split('(')[0]}**")

    if uploaded_file:
        if st.button("✨ Generate Caption ✨", use_container_width=True):

            with st.spinner("🧠 AI is analyzing pixels..."):
                try:
                    # 1. Gemini Logic
                    if "Gemini" in engine:
                        try:
                            caption = generate_gemini(image)
                            method_label = "Google Gemini 1.5 Pro"
                        except Exception as e:
                            st.error("⚠️ Network Error!")
                            st.warning("Check VPN (Port 7892)")
                            st.stop()

                    # 2. Local AI Logic
                    else:
                        tokenizer = load_resources()
                        if tokenizer:
                            feat_extractor, caption_model, pp_func, t_size = load_local_model(model_choice)
                            if caption_model:
                                img_array = img_to_array(image.resize(t_size))
                                img_array = img_array.reshape(
                                    (1, img_array.shape[0], img_array.shape[1], img_array.shape[2]))
                                img_array = pp_func(img_array)

                                feature = feat_extractor.predict(img_array, verbose=0)

                                if "Beam" in search_mode:
                                    caption = beam_search(caption_model, feature, tokenizer, k=beam_k)
                                    method_label = f"Local AI ({model_choice}) + Beam"
                                else:
                                    caption = greedy_search(caption_model, feature, tokenizer)
                                    method_label = f"Local AI ({model_choice}) + Greedy"
                            else:
                                st.error("Model file missing.")
                                st.stop()
                        else:
                            st.error("Tokenizer missing.")
                            st.stop()

                    # --- FINAL RESULT ---
                    st.markdown(f"""
                    <div class="result-card">
                        <span class="badge">{method_label}</span>
                        <hr style="border: 0; border-top: 1px solid #eee; margin: 15px 0;">
                        <div class="caption-text">“{caption.strip().title()}”</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # 🎈🎈 BALLOONS ANIMATION 🎈🎈
                    st.balloons()

                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.info("Waiting for image upload...")

# Footer
st.markdown("---")
st.markdown('<div class="dev-credit">© 2025 AI Project | Developed by Golam Rohman Ruhit</div>', unsafe_allow_html=True)