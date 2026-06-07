# ================================================================
#  🌸  IRIS FLOWER CLASSIFICATION — STREAMLIT WEB APP
#  Project 2 | DecodeLabs AI Industrial Training · Batch 2026
#  Developer : Mowshik | KPRIET
#  Algorithm : K-Nearest Neighbors (KNN)
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import time
import io
import datetime
import os

# ── Page config MUST be first Streamlit call ─────────────────
st.set_page_config(
    page_title="Iris Flower AI",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ================================================================
# §1  IMAGE PATHS  ← Paste your actual image paths here
# ================================================================
SETOSA_IMAGE     = "https://raw.githubusercontent.com/Mowshik1210/Decodelabs_Iris-AI/main/Assests/Iris-setosa.jpg"   # ← Update path
VERSICOLOR_IMAGE = "https://raw.githubusercontent.com/Mowshik1210/Decodelabs_Iris-AI/main/Assests/iris Versicolor.jpg"  # ← Update path
VIRGINICA_IMAGE  = "https://raw.githubusercontent.com/Mowshik1210/Decodelabs_Iris-AI/main/Assests/iris Virginica.jpg"   # ← Update path

SPECIES_META = {
    "Iris Setosa": {
        "image"  : SETOSA_IMAGE,
        "emoji"  : "🌷",
        "color"  : "#e74c7c",
        "accent" : "#fce4ec",
        "latin"  : "Iris setosa",
        "origin" : "Arctic & subarctic regions",
        "trait"  : "Smallest petals · Very distinct",
        "desc"   : (
            "The most easily identifiable species. "
            "Known for its remarkably small petal size, "
            "making it linearly separable from the others."
        ),
    },
    "Iris Versicolor": {
        "image"  : VERSICOLOR_IMAGE,
        "emoji"  : "🌺",
        "color"  : "#0288d1",
        "accent" : "#e1f5fe",
        "latin"  : "Iris versicolor",
        "origin" : "Eastern North America",
        "trait"  : "Medium size · Blue-violet blooms",
        "desc"   : (
            "The Blue Flag Iris, native to North America. "
            "Displays medium-range sepal and petal measurements, "
            "sometimes overlapping with Virginica."
        ),
    },
    "Iris Virginica": {
        "image"  : VIRGINICA_IMAGE,
        "emoji"  : "💐",
        "color"  : "#7b1fa2",
        "accent" : "#f3e5f5",
        "latin"  : "Iris virginica",
        "origin" : "Eastern United States",
        "trait"  : "Largest petals · Deep violet blooms",
        "desc"   : (
            "The Virginia Iris — largest species in the dataset. "
            "Features the biggest petals and often the longest sepals, "
            "thriving in coastal wetlands."
        ),
    },
}

# ================================================================
# §2  CUSTOM CSS — BOTANICAL LUXURY THEME
# ================================================================
def inject_css():
    st.markdown(
        """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600;700&display=swap');

/* ── CSS Variables ── */
:root {
  --forest   : #1a3d26;
  --sage     : #3d7a56;
  --mint     : #6abf8a;
  --lime     : #a8d5b5;
  --cream    : #faf7f0;
  --warm     : #f5efe6;
  --glass    : rgba(255,255,255,0.72);
  --glass-b  : rgba(255,255,255,0.90);
  --shadow   : rgba(26, 61, 38, 0.12);
  --shadow-d : rgba(26, 61, 38, 0.22);
  --text     : #1c2e22;
  --text-m   : #3d5a47;
  --text-l   : #6b8f77;
  --radius   : 20px;
  --radius-s : 12px;
}

/* ── Global Reset ── */
* { box-sizing: border-box; margin: 0; padding: 0; }

/* ── App Background ── */
.stApp {
  background:
    radial-gradient(ellipse at 0% 0%, rgba(106,191,138,0.18) 0%, transparent 55%),
    radial-gradient(ellipse at 100% 100%, rgba(168,213,181,0.15) 0%, transparent 55%),
    linear-gradient(160deg, #f0faf4 0%, #faf7f0 50%, #eef7f0 100%);
  font-family: 'DM Sans', sans-serif;
  color: var(--text);
}

/* ── Hide Streamlit Chrome ── */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"]  { visibility: hidden; height: 0; }
.block-container {
  padding-top: 1rem !important;
  padding-bottom: 2rem !important;
  max-width: 1240px !important;
}

/* ================================================================
   HERO BANNER
================================================================ */
.hero-wrap {
  background: linear-gradient(135deg,
    #0d2b18 0%, #1a3d26 20%, #2d6642 45%,
    #3d7a56 70%, #4a9668 100%);
  border-radius: 28px;
  padding: 3.5rem 2rem 3rem;
  text-align: center;
  position: relative;
  overflow: hidden;
  margin-bottom: 2.5rem;
  box-shadow: 0 24px 72px rgba(13,43,24,0.4), 0 4px 16px rgba(0,0,0,0.1);
}

/* Floating botanical orbs */
.hero-wrap::before {
  content: '';
  position: absolute;
  top: -80px; right: -80px;
  width: 320px; height: 320px;
  background: radial-gradient(circle, rgba(106,191,138,0.18) 0%, transparent 70%);
  border-radius: 50%;
  animation: floatOrb 8s ease-in-out infinite;
}
.hero-wrap::after {
  content: '';
  position: absolute;
  bottom: -60px; left: -60px;
  width: 240px; height: 240px;
  background: radial-gradient(circle, rgba(168,213,181,0.12) 0%, transparent 70%);
  border-radius: 50%;
  animation: floatOrb 10s ease-in-out infinite reverse;
}

@keyframes floatOrb {
  0%, 100% { transform: translate(0,0) scale(1); }
  50%       { transform: translate(20px,-20px) scale(1.1); }
}

.hero-leaf-bg {
  position: absolute; inset: 0;
  background-image:
    radial-gradient(circle at 20% 80%, rgba(255,255,255,0.04) 0%, transparent 40%),
    radial-gradient(circle at 80% 20%, rgba(255,255,255,0.04) 0%, transparent 40%);
  pointer-events: none;
}

.hero-emoji-row {
  font-size: 2.4rem;
  margin-bottom: 1rem;
  animation: floatEmoji 4s ease-in-out infinite;
  letter-spacing: 0.5rem;
  display: block;
  position: relative; z-index: 1;
}
@keyframes floatEmoji {
  0%, 100% { transform: translateY(0);    }
  50%       { transform: translateY(-8px); }
}

.hero-title {
  font-family: 'Playfair Display', serif;
  font-size: clamp(1.8rem, 4vw, 3rem);
  font-weight: 800;
  color: #ffffff;
  line-height: 1.2;
  letter-spacing: -0.5px;
  text-shadow: 0 2px 24px rgba(0,0,0,0.25);
  margin-bottom: 0.6rem;
  position: relative; z-index: 1;
}

.hero-subtitle {
  font-size: 1rem;
  color: rgba(255,255,255,0.78);
  font-weight: 400;
  margin-bottom: 1.8rem;
  position: relative; z-index: 1;
  letter-spacing: 0.3px;
}

.hero-badges {
  display: flex;
  justify-content: center;
  gap: 0.75rem;
  flex-wrap: wrap;
  position: relative; z-index: 1;
}

.badge {
  background: rgba(255,255,255,0.12);
  backdrop-filter: blur(12px);
  border: 1px solid rgba(255,255,255,0.25);
  color: rgba(255,255,255,0.92);
  padding: 0.4rem 1.1rem;
  border-radius: 100px;
  font-size: 0.8rem;
  font-weight: 500;
  letter-spacing: 0.3px;
  transition: background 0.3s;
}
.badge:hover { background: rgba(255,255,255,0.22); }

/* ================================================================
   GLASS CARDS — UNIVERSAL
================================================================ */
.glass-card {
  background: var(--glass);
  backdrop-filter: blur(24px);
  -webkit-backdrop-filter: blur(24px);
  border: 1px solid rgba(255,255,255,0.85);
  border-radius: var(--radius);
  padding: 1.8rem;
  margin-bottom: 1.8rem;
  box-shadow:
    0 8px 32px var(--shadow),
    0 2px 8px rgba(0,0,0,0.04),
    inset 0 1px 0 rgba(255,255,255,0.9);
  transition: transform 0.3s cubic-bezier(.25,.8,.25,1),
              box-shadow 0.3s ease;
}
.glass-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 20px 56px var(--shadow-d), 0 4px 16px rgba(0,0,0,0.06);
}

/* ================================================================
   SECTION TITLES
================================================================ */
.sec-title {
  font-family: 'Playfair Display', serif;
  font-size: 1.55rem;
  font-weight: 700;
  color: var(--forest);
  margin-bottom: 1.2rem;
  display: flex;
  align-items: center;
  gap: 0.55rem;
  letter-spacing: -0.3px;
}
.sec-title .icon { font-size: 1.4rem; }

.sec-desc {
  font-size: 0.92rem;
  color: var(--text-m);
  line-height: 1.75;
  margin-bottom: 1.4rem;
}

/* ================================================================
   SPECIES CARDS
================================================================ */
.species-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1.1rem;
}
@media (max-width: 700px) {
  .species-grid { grid-template-columns: 1fr; }
}

.species-card {
  background: rgba(255,255,255,0.82);
  border-radius: 16px;
  padding: 1.5rem 1.2rem;
  text-align: center;
  border: 2px solid transparent;
  transition: all 0.35s cubic-bezier(.25,.8,.25,1);
  position: relative;
  overflow: hidden;
  cursor: default;
}
.species-card::after {
  content: '';
  position: absolute;
  bottom: 0; left: 0; right: 0;
  height: 4px;
  background: var(--bar-color, #3d7a56);
  transform: scaleX(0);
  transition: transform 0.35s ease;
  transform-origin: left;
}
.species-card:hover::after   { transform: scaleX(1); }
.species-card:hover          { border-color: var(--bar-color, #3d7a56); transform: translateY(-6px);
  box-shadow: 0 16px 40px rgba(0,0,0,0.1); }

.sp-emoji  { font-size: 2.8rem; display: block; margin-bottom: 0.6rem; }
.sp-name   { font-family: 'Playfair Display', serif; font-size: 1.15rem;
             font-weight: 700; color: var(--forest); margin-bottom: 0.15rem; }
.sp-latin  { font-style: italic; font-size: 0.75rem; color: var(--text-l); margin-bottom: 0.3rem; }
.sp-origin { font-size: 0.75rem; color: var(--text-l); margin-bottom: 0.6rem; }
.sp-trait  { font-size: 0.78rem; font-weight: 600; padding: 0.25rem 0.75rem;
             border-radius: 100px; display: inline-block; margin-bottom: 0.7rem; }
.sp-desc   { font-size: 0.82rem; color: var(--text-m); line-height: 1.6; }

/* ================================================================
   FEATURE DESCRIPTION CARDS
================================================================ */
.feature-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
  margin-top: 0.5rem;
}
.feat-card {
  background: linear-gradient(135deg,
    rgba(240,250,244,0.9) 0%, rgba(235,248,245,0.9) 100%);
  border-radius: var(--radius-s);
  padding: 1.1rem 1.2rem;
  border-left: 4px solid var(--left-col, #6abf8a);
  display: flex; align-items: flex-start; gap: 0.8rem;
}
.feat-icon  { font-size: 1.6rem; flex-shrink: 0; margin-top: 2px; }
.feat-label { font-weight: 700; color: var(--forest); font-size: 0.9rem; margin-bottom: 0.15rem; }
.feat-range { font-size: 0.76rem; font-weight: 600; color: var(--sage); margin-bottom: 0.2rem; }
.feat-desc  { font-size: 0.8rem; color: var(--text-m); line-height: 1.55; }

/* ================================================================
   SLIDER OVERRIDES
================================================================ */
[data-testid="stSlider"] > div > div > div > div {
  background: linear-gradient(90deg, #3d7a56, #6abf8a) !important;
}
[data-testid="stSlider"] > div > div > div > div > div {
  border: 2px solid #3d7a56 !important;
  box-shadow: 0 2px 8px rgba(61,122,86,0.3) !important;
}
.slider-wrap { margin-bottom: 0.5rem; }
.slider-header {
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 2px;
}
.slider-label { font-size: 0.88rem; font-weight: 600; color: var(--forest); }
.slider-val   {
  font-size: 0.82rem; font-weight: 700;
  background: linear-gradient(135deg, #1a3d26, #3d7a56);
  color: white; padding: 0.2rem 0.65rem;
  border-radius: 100px;
}

/* ================================================================
   PREDICT BUTTON
================================================================ */
.stButton > button {
  background: linear-gradient(135deg, #1a3d26 0%, #2d6642 40%, #3d7a56 70%, #6abf8a 100%) !important;
  color: white !important;
  border: none !important;
  border-radius: 100px !important;
  padding: 0.85rem 2.5rem !important;
  font-family: 'DM Sans', sans-serif !important;
  font-size: 1.05rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.5px !important;
  box-shadow: 0 8px 28px rgba(26,61,38,0.35) !important;
  transition: all 0.3s cubic-bezier(.25,.8,.25,1) !important;
  width: 100% !important;
  position: relative !important;
}
.stButton > button:hover {
  transform: translateY(-3px) !important;
  box-shadow: 0 14px 40px rgba(26,61,38,0.45) !important;
  background: linear-gradient(135deg, #0d2b18 0%, #1a3d26 40%, #2d6642 70%, #4a9668 100%) !important;
}
.stButton > button:active {
  transform: translateY(0) !important;
  box-shadow: 0 4px 14px rgba(26,61,38,0.3) !important;
}

/* ================================================================
   RESULT CARD
================================================================ */
.result-card {
  border-radius: var(--radius);
  padding: 2rem 1.8rem;
  text-align: center;
  animation: riseIn 0.6s cubic-bezier(.34,1.56,.64,1) both;
  position: relative;
  overflow: hidden;
}
@keyframes riseIn {
  from { opacity: 0; transform: translateY(28px) scale(0.96); }
  to   { opacity: 1; transform: translateY(0)    scale(1);    }
}

.result-checkmark {
  font-size: 2.8rem;
  display: block;
  margin-bottom: 0.5rem;
  animation: popIn 0.5s 0.3s cubic-bezier(.34,1.56,.64,1) both;
}
@keyframes popIn {
  from { transform: scale(0) rotate(-15deg); opacity: 0; }
  to   { transform: scale(1) rotate(0);      opacity: 1; }
}

.result-label   { font-size: 0.8rem; font-weight: 600; text-transform: uppercase;
                  letter-spacing: 1.5px; opacity: 0.65; margin-bottom: 0.4rem; }
.result-species { font-family: 'Playfair Display', serif; font-size: 2rem;
                  font-weight: 800; margin-bottom: 0.6rem; }
.result-conf    { font-size: 0.95rem; font-weight: 600; opacity: 0.8; margin-bottom: 1.2rem; }

.conf-track {
  background: rgba(0,0,0,0.08);
  border-radius: 100px; height: 8px;
  overflow: hidden; margin: 0 auto 1.5rem;
  max-width: 280px;
}
.conf-fill {
  height: 8px;
  border-radius: 100px;
  background: linear-gradient(90deg, var(--fill-from, #3d7a56), var(--fill-to, #6abf8a));
  animation: growBar 1s 0.5s ease both;
}
@keyframes growBar {
  from { width: 0 !important; }
}

.prob-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 0.6rem;
  margin-top: 1rem;
}
.prob-chip {
  background: rgba(255,255,255,0.55);
  border-radius: 10px;
  padding: 0.5rem;
  font-size: 0.75rem;
  text-align: center;
}
.prob-chip .pct { font-weight: 700; font-size: 0.95rem; display: block; }
.prob-chip .nm  { opacity: 0.65; }

/* ================================================================
   IMAGE CARD
================================================================ */
.img-card {
  background: rgba(255,255,255,0.8);
  border-radius: var(--radius);
  overflow: hidden;
  box-shadow: 0 12px 36px var(--shadow-d);
  margin-top: 1.2rem;
  animation: riseIn 0.7s 0.2s cubic-bezier(.34,1.56,.64,1) both;
}
.img-card .img-caption {
  padding: 0.8rem;
  text-align: center;
  font-size: 0.82rem;
  color: var(--text-m);
  font-style: italic;
  background: var(--warm);
}
.img-placeholder {
  background: linear-gradient(135deg, #e8f5e9, #e0f7fa);
  padding: 3rem 1rem;
  text-align: center;
  font-size: 5rem;
}

/* ================================================================
   STATS SECTION
================================================================ */
.stats-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 1rem;
}
@media (max-width: 700px) { .stats-grid { grid-template-columns: repeat(2,1fr); } }

.stat-chip {
  background: rgba(255,255,255,0.78);
  border-radius: var(--radius-s);
  padding: 1.2rem;
  text-align: center;
  border: 1px solid rgba(168,213,181,0.5);
  transition: transform 0.2s;
}
.stat-chip:hover   { transform: translateY(-3px); }
.stat-chip .sv     { font-family: 'Playfair Display', serif;
                     font-size: 2rem; font-weight: 800; color: var(--forest); }
.stat-chip .sl     { font-size: 0.75rem; color: var(--text-l); margin-top: 0.2rem; font-weight: 500; }

/* ================================================================
   KNN EXPLAINER
================================================================ */
.knn-steps {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px,1fr));
  gap: 1rem;
  margin-top: 0.5rem;
}
.knn-step {
  background: rgba(255,255,255,0.75);
  border-radius: var(--radius-s);
  padding: 1.1rem;
  border-top: 3px solid var(--step-col, #6abf8a);
  text-align: center;
}
.step-num  { font-family: 'Playfair Display', serif; font-size: 1.8rem;
             font-weight: 800; color: var(--forest); }
.step-lbl  { font-weight: 700; color: var(--text); font-size: 0.88rem; margin: 0.3rem 0 0.25rem; }
.step-desc { font-size: 0.78rem; color: var(--text-m); line-height: 1.5; }

/* ================================================================
   HISTORY TABLE
================================================================ */
[data-testid="stDataFrame"] {
  border-radius: var(--radius-s) !important;
  overflow: hidden !important;
}

/* ================================================================
   DOWNLOAD BUTTON
================================================================ */
[data-testid="stDownloadButton"] > button {
  background: linear-gradient(135deg, #f0faf4, #e8f5e9) !important;
  color: var(--forest) !important;
  border: 2px solid #a8d5b5 !important;
  border-radius: 100px !important;
  font-weight: 700 !important;
  font-family: 'DM Sans', sans-serif !important;
  transition: all 0.25s ease !important;
  padding: 0.6rem 1.5rem !important;
}
[data-testid="stDownloadButton"] > button:hover {
  background: linear-gradient(135deg, #e8f5e9, #d0edda) !important;
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 20px rgba(26,61,38,0.15) !important;
}

/* ================================================================
   FOOTER
================================================================ */
.footer-wrap {
  background: linear-gradient(135deg, #0d2b18 0%, #1a3d26 50%, #2d6642 100%);
  border-radius: var(--radius);
  padding: 2.2rem 2rem;
  text-align: center;
  margin-top: 2.5rem;
  position: relative;
  overflow: hidden;
}
.footer-wrap::before {
  content: '🌿🌱🍃🌸🌿🌱🍃🌸🌿🌱🍃🌸';
  position: absolute; top: 12px; left: 0; right: 0;
  font-size: 0.8rem; opacity: 0.2; letter-spacing: 0.3rem;
}
.footer-brand  { font-family: 'Playfair Display', serif; font-size: 1.4rem;
                 font-weight: 800; color: white; margin-bottom: 0.4rem; }
.footer-sub    { font-size: 0.82rem; color: rgba(255,255,255,0.65); margin-bottom: 1.2rem; }
.footer-chips  { display: flex; justify-content: center; flex-wrap: wrap; gap: 0.6rem; }
.footer-chip   { background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2);
                 color: rgba(255,255,255,0.85); padding: 0.3rem 0.9rem;
                 border-radius: 100px; font-size: 0.78rem; }
.footer-copy   { font-size: 0.75rem; color: rgba(255,255,255,0.4);
                 margin-top: 1.2rem; }

/* ================================================================
   EXPANDER STYLE
================================================================ */
[data-testid="stExpander"] > details > summary {
  background: rgba(240,250,244,0.8);
  border-radius: var(--radius-s) !important;
  font-family: 'DM Sans', sans-serif !important;
  font-weight: 600 !important;
  color: var(--forest) !important;
  padding: 0.9rem 1.2rem !important;
  border: 1px solid rgba(168,213,181,0.6) !important;
}
[data-testid="stExpander"] > details > summary:hover {
  background: rgba(232,245,233,0.9) !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #3d7a56 !important; }
</style>
        """,
        unsafe_allow_html=True,
    )


# ================================================================
# §3  MODEL LOADING  ← Your KNN model is trained here
# ================================================================
@st.cache_resource(show_spinner="🌱 Growing your model...")
def load_model():
    """
    Trains and caches the KNN model.
    ── To use a pre-saved model, replace this block with: ──
        import joblib
        model  = joblib.load("model/knn_model.pkl")
        scaler = joblib.load("model/scaler.pkl")
    ────────────────────────────────────────────────────────
    """
    # ── Paste your iris_knn.py training code below ──────────
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import (
        accuracy_score, confusion_matrix, classification_report
    )

    iris = load_iris()
    X    = iris.data
    y    = iris.target

    # Feature scaling
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # KNN — k=5
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    # Evaluation metrics (stored but not printed to UI)
    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm       = confusion_matrix(y_test, y_pred)
    report   = classification_report(y_test, y_pred, target_names=iris.target_names)
    # ── End of your training code ────────────────────────────

    return model, scaler, iris, accuracy, cm, report


# ================================================================
# §4  PREDICTION LOGIC
# ================================================================
def predict_species(model, scaler, iris, sepal_l, sepal_w, petal_l, petal_w):
    """Scale inputs → KNN predict → return name, confidence, all probs."""
    raw_features = np.array([[sepal_l, sepal_w, petal_l, petal_w]])
    scaled       = scaler.transform(raw_features)

    pred_idx   = model.predict(scaled)[0]
    pred_proba = model.predict_proba(scaled)[0]

    raw_name     = iris.target_names[pred_idx]
    species_name = f"Iris {raw_name.capitalize()}"
    confidence   = pred_proba[pred_idx] * 100

    all_probs = {
        f"Iris {iris.target_names[i].capitalize()}": round(pred_proba[i] * 100, 1)
        for i in range(len(iris.target_names))
    }
    return species_name, confidence, all_probs


# ================================================================
# §5  DISPLAY RESULT
# ================================================================
def display_result(species_name: str, confidence: float, all_probs: dict):
    """Render the animated prediction result card + image."""
    meta   = SPECIES_META.get(species_name, {})
    color  = meta.get("color",  "#3d7a56")
    accent = meta.get("accent", "#e8f5e9")
    emoji  = meta.get("emoji",  "🌸")

    # ── Result Card ──────────────────────────────────────────
    prob_chips_html = "".join(
        f"""<div class="prob-chip">
              <span class="pct" style="color:{SPECIES_META.get(sp,{}).get('color','#3d7a56')}">
                {pct}%
              </span>
              <span class="nm">{sp.replace('Iris ','')}</span>
            </div>"""
        for sp, pct in all_probs.items()
    )

    st.markdown(
        f"""
        <div class="result-card" style="background:linear-gradient(135deg,{accent},{accent}cc);
             border:2px solid {color}33;">
          <span class="result-checkmark">✅</span>
          <div class="result-label" style="color:{color}">Predicted Species</div>
          <div class="result-species" style="color:{color}">{emoji} {species_name}</div>
          <div class="result-conf">Confidence · {confidence:.1f}%</div>
          <div class="conf-track">
            <div class="conf-fill"
                 style="width:{confidence:.1f}%;
                        --fill-from:{color};
                        --fill-to:{color}aa;">
            </div>
          </div>
          <div style="font-size:.78rem;color:#6b8f77;margin-bottom:.6rem;font-weight:500;">
            Probability Distribution
          </div>
          <div class="prob-grid">{prob_chips_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

      # ── Flower Image Card ──────────────────────────────────────
    img_path = meta.get("image", "")
    st.markdown('<div class="img-card">', unsafe_allow_html=True)

    if img_path:
        # ✅ Detect URL vs local path
        is_url = img_path.startswith("http://") or img_path.startswith("https://")

        if is_url:
          try:
            st.image(img_path, use_container_width=True)   # ← handles GitHub URLs directly
          except Exception:
            st.markdown(f'<div class="img-placeholder">{emoji}</div>', unsafe_allow_html=True)

        elif os.path.exists(img_path):
          try:
            from PIL import Image
            st.image(Image.open(img_path), use_container_width=True)
          except Exception:
            st.markdown(f'<div class="img-placeholder">{emoji}</div>', unsafe_allow_html=True)

        else:
          st.markdown(f'<div class="img-placeholder">{emoji}</div>', unsafe_allow_html=True)

          st.markdown(
          f'<div class="img-caption">🔬 {meta.get("latin","—")} · {meta.get("origin","—")}</div>',
          unsafe_allow_html=True,
       )
        st.markdown("</div>", unsafe_allow_html=True)


# ================================================================
# §6  HERO SECTION
# ================================================================
def render_hero():
    st.markdown(
        """
        <div class="hero-wrap">
          <div class="hero-leaf-bg"></div>
          <span class="hero-emoji-row">🌸 🌿 🌺 🍃 💐</span>
          <div class="hero-title">IRIS FLOWER AI</div>
          <div class="hero-subtitle">
            Iris flower classification using AI by MOWSHIK G /
            CSE(AI&ML) STUDENT at KPRIET
          </div>
          <div class="hero-badges">
            <span class="badge">🤖 K-Nearest Neighbors</span>
            <span class="badge">🧬 Scikit-Learn</span>
            <span class="badge">📊 150 Samples · 3 Classes</span>
            <span class="badge">⚗️ Feature Scaling · StandardScaler</span>
            <span class="badge">🌱 Eco-Friendly AI</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ================================================================
# §7  ABOUT — DATASET + SPECIES
# ================================================================
def render_about():
    with st.container():
        st.markdown(
            """
            <div class="glass-card">
              <div class="sec-title">
                <span class="icon">🌿</span> The Iris Dataset
              </div>
              <div class="sec-desc">
                Introduced by statistician Ronald Fisher in 1936, the Iris dataset is one of the
                most celebrated benchmarks in machine learning. It contains
                <strong>150 flower samples</strong> across <strong>3 species</strong>,
                each described by 4 physical measurements — sepal length, sepal width,
                petal length, and petal width — all in centimetres.
                KNN is especially effective here: it predicts a new flower's species
                by finding the <em>k</em> most similar known samples and voting on the result.
              </div>
            """,
            unsafe_allow_html=True,
        )

        species_cards = ""
        styles = [
            ("--bar-color:#e74c7c", "#e74c7c", "#fce4ec"),
            ("--bar-color:#0288d1", "#0288d1", "#e1f5fe"),
            ("--bar-color:#7b1fa2", "#7b1fa2", "#f3e5f5"),
        ]
        for (name, meta), (style, col, bg) in zip(SPECIES_META.items(), styles):
            species_cards += f"""
            <div class="species-card" style="{style}; background:{bg}88;">
              <span class="sp-emoji">{meta['emoji']}</span>
              <div class="sp-name">{name}</div>
              <div class="sp-latin">{meta['latin']}</div>
              <div class="sp-origin">📍 {meta['origin']}</div>
              <span class="sp-trait"
                style="background:{bg};color:{col};border:1px solid {col}44;">
                {meta['trait']}
              </span>
              <div class="sp-desc">{meta['desc']}</div>
            </div>"""

        st.markdown(
            f'<div class="species-grid">{species_cards}</div></div>',
            unsafe_allow_html=True,
        )


# ================================================================
# §8  FEATURE DESCRIPTION CARDS
# ================================================================
def render_feature_cards():
    st.markdown(
        """
        <div class="glass-card">
          <div class="sec-title"><span class="icon">🔬</span> Measurement Features</div>
          <div class="feature-grid">
            <div class="feat-card" style="--left-col:#3d7a56">
              <span class="feat-icon">📏</span>
              <div>
                <div class="feat-label">Sepal Length</div>
                <div class="feat-range">Range: 4.3 – 7.9 cm</div>
                <div class="feat-desc">
                  The outer green leaf-like parts that protect the flower bud.
                  Setosa has the shortest sepals; Virginica the longest.
                </div>
              </div>
            </div>
            <div class="feat-card" style="--left-col:#0288d1">
              <span class="feat-icon">↔️</span>
              <div>
                <div class="feat-label">Sepal Width</div>
                <div class="feat-range">Range: 2.0 – 4.4 cm</div>
                <div class="feat-desc">
                  Surprisingly, Setosa has the widest sepals despite its compact size.
                  A key discriminating dimension in the dataset.
                </div>
              </div>
            </div>
            <div class="feat-card" style="--left-col:#e74c7c">
              <span class="feat-icon">🌸</span>
              <div>
                <div class="feat-label">Petal Length</div>
                <div class="feat-range">Range: 1.0 – 6.9 cm</div>
                <div class="feat-desc">
                  The most powerful discriminating feature. Setosa petals are
                  dramatically shorter, making them instantly separable.
                </div>
              </div>
            </div>
            <div class="feat-card" style="--left-col:#7b1fa2">
              <span class="feat-icon">🌼</span>
              <div>
                <div class="feat-label">Petal Width</div>
                <div class="feat-range">Range: 0.1 – 2.5 cm</div>
                <div class="feat-desc">
                  Second most discriminative feature after petal length.
                  Together these two petal dimensions drive most of KNN's accuracy.
                </div>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ================================================================
# §9  INPUT SLIDERS
# ================================================================
def render_inputs():
    st.markdown(
        """
        <div class="sec-title" style="margin-bottom:.8rem;">
          <span class="icon">🎛️</span> Enter Flower Measurements
        </div>
        """,
        unsafe_allow_html=True,
    )

    sliders = [
        ("Sepal Length", "sepal_l", 4.3, 7.9, 5.8, "📏"),
        ("Sepal Width",  "sepal_w", 2.0, 4.4, 3.0, "↔️"),
        ("Petal Length", "petal_l", 1.0, 6.9, 3.7, "🌸"),
        ("Petal Width",  "petal_w", 0.1, 2.5, 1.2, "🌼"),
    ]

    values = {}
    for label, key, mn, mx, default, icon in sliders:
        st.markdown(
            f"""<div class="slider-header">
                  <span class="slider-label">{icon} {label} (cm)</span>
                </div>""",
            unsafe_allow_html=True,
        )
        val = st.slider(
            label, mn, mx, default,
            step=0.1, label_visibility="collapsed", key=key
        )
        st.markdown(
            f'<div style="text-align:right;margin-top:-10px;margin-bottom:8px;">'
            f'<span class="slider-val">{val:.1f} cm</span></div>',
            unsafe_allow_html=True,
        )
        values[key] = val

    return (
        values["sepal_l"], values["sepal_w"],
        values["petal_l"], values["petal_w"],
    )


# ================================================================
# §10  DATASET STATISTICS
# ================================================================
def render_stats(iris):
    df     = pd.DataFrame(iris.data, columns=iris.feature_names)
    labels = [f"Iris {n.capitalize()}" for n in iris.target_names]

    st.markdown(
        f"""
        <div class="glass-card">
          <div class="sec-title"><span class="icon">📊</span> Dataset Statistics</div>
          <div class="stats-grid">
            <div class="stat-chip">
              <div class="sv">150</div>
              <div class="sl">Total Samples</div>
            </div>
            <div class="stat-chip">
              <div class="sv">4</div>
              <div class="sl">Features</div>
            </div>
            <div class="stat-chip">
              <div class="sv">3</div>
              <div class="sl">Species Classes</div>
            </div>
            <div class="stat-chip">
              <div class="sv">50</div>
              <div class="sl">Samples / Class</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("📈 View Detailed Feature Statistics"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Descriptive Summary")
            st.dataframe(df.describe().round(2), use_container_width=True)
        with col2:
            st.subheader("Samples per Species")
            counts = pd.Series(iris.target).map(
                {i: labels[i] for i in range(3)}
            ).value_counts()
            st.dataframe(
                counts.reset_index().rename(columns={"index": "Species", 0: "Count"}),
                use_container_width=True,
            )


# ================================================================
# §11  HOW KNN WORKS EXPLAINER
# ================================================================
def render_knn_explainer():
    with st.expander("🧠  How Does KNN Work? — Interactive Explainer"):
        st.markdown(
            """
            <div style="margin-bottom:1rem;">
              <div class="sec-desc" style="margin-bottom:1rem;">
                <strong>K-Nearest Neighbors (KNN)</strong> is a lazy, instance-based learning
                algorithm. It stores all training examples and classifies new data by majority
                vote of its <em>k</em> nearest neighbours in feature space.
                With <em>k = 5</em> and StandardScaler normalisation this model reaches
                <strong>~97% accuracy</strong> on the Iris dataset.
              </div>
            </div>
            <div class="knn-steps">
              <div class="knn-step" style="--step-col:#3d7a56">
                <div class="step-num">①</div>
                <div class="step-lbl">Scale Features</div>
                <div class="step-desc">
                  StandardScaler removes the mean and scales to unit variance so
                  all features contribute equally to distance.
                </div>
              </div>
              <div class="knn-step" style="--step-col:#0288d1">
                <div class="step-num">②</div>
                <div class="step-lbl">Compute Distance</div>
                <div class="step-desc">
                  Euclidean distance is calculated between the new point and
                  every point in the training set.
                </div>
              </div>
              <div class="knn-step" style="--step-col:#e74c7c">
                <div class="step-num">③</div>
                <div class="step-lbl">Find k = 5 Neighbours</div>
                <div class="step-desc">
                  The 5 training points with smallest distance to the query
                  are selected as neighbours.
                </div>
              </div>
              <div class="knn-step" style="--step-col:#7b1fa2">
                <div class="step-num">④</div>
                <div class="step-lbl">Majority Vote</div>
                <div class="step-desc">
                  Each neighbour casts one vote for its species label.
                  The class with the most votes wins — that's the prediction.
                </div>
              </div>
              <div class="knn-step" style="--step-col:#f57c00">
                <div class="step-num">⑤</div>
                <div class="step-lbl">Confidence Score</div>
                <div class="step-desc">
                  Confidence = fraction of neighbours that voted for the winning class.
                  E.g., 5/5 = 100%, 4/5 = 80%.
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ================================================================
# §12  PREDICTION HISTORY
# ================================================================
def render_history():
    if not st.session_state.get("history"):
        return

    st.markdown(
        """
        <div class="glass-card">
          <div class="sec-title">
            <span class="icon">📋</span> Prediction History
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df_hist = pd.DataFrame(st.session_state.history)
    st.dataframe(df_hist, use_container_width=True, hide_index=True)

    if st.button("🗑️  Clear History", use_container_width=True):
        st.session_state.history = []
        if "last_prediction" in st.session_state:
            del st.session_state["last_prediction"]
        st.rerun()


# ================================================================
# §13  MODEL PERFORMANCE REPORT
# ================================================================
def render_model_report(accuracy: float, cm, report: str):
    with st.expander(f"🏆  Model Performance Report — Accuracy: {accuracy*100:.1f}%"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Confusion Matrix")
            labels = ["Setosa", "Versicolor", "Virginica"]
            cm_df  = pd.DataFrame(cm, index=labels, columns=labels)
            st.dataframe(cm_df, use_container_width=True)
        with col2:
            st.subheader("Classification Report")
            st.expander(report)


# ================================================================
# §14  FOOTER
# ================================================================
def render_footer(accuracy: float):
    st.markdown(
        f"""
        <div class="footer-wrap">
          <div class="footer-brand">🌸 Iris Flower AI</div>
          <div class="footer-sub">
            An AI for classifying iris flowers based on their measurements 
          </div>
          <div class="footer-chips">
            <span class="footer-chip">👨‍💻 Developer: Mowshik G</span>
            <span class="footer-chip">🏫 KPRIET student</span>
            <span class="footer-chip">🤖 Algorithm: KNN (k=5)</span>
            <span class="footer-chip">🎯 Model Accuracy: {accuracy*100:.1f}%</span>
            <span class="footer-chip">📦 Scikit-Learn · Streamlit · Pandas</span>
          </div>
          <div class="footer-copy">
            Thank you for using Iris Flower AI! · Nature-Inspired AI ·
            © 2026 Mowshik G / 💚🤍💚🤍
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ================================================================
# §15  MAIN
# ================================================================
def main():
    inject_css()

    # ── Session State ───────────────────────────────────────
    if "history" not in st.session_state:
        st.session_state.history = []

    # ── Load model ──────────────────────────────────────────
    model, scaler, iris, accuracy, cm, report = load_model()

    # ── Hero ────────────────────────────────────────────────
    render_hero()

    # ── About + Species ─────────────────────────────────────
    render_about()

    # ── Feature Cards ───────────────────────────────────────
    render_feature_cards()

    # ── Main: Inputs (left) + Results (right) ───────────────
    col_in, col_res = st.columns([1, 1], gap="large")

    with col_in:
        with st.container():
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            sepal_l, sepal_w, petal_l, petal_w = render_inputs()

            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("🔬 Predict Flower Species", use_container_width=True):
                with st.spinner("🌱 Analysing your measurements…"):
                    time.sleep(0.9)   # satisfying UX pause

                species, confidence, all_probs = predict_species(
                    model, scaler, iris, sepal_l, sepal_w, petal_l, petal_w
                )

                st.session_state.last_prediction = {
                    "species"   : species,
                    "confidence": confidence,
                    "all_probs" : all_probs,
                }
                st.session_state.history.append({
                    "Time"      : datetime.datetime.now().strftime("%H:%M:%S"),
                    "Sepal L"   : sepal_l,
                    "Sepal W"   : sepal_w,
                    "Petal L"   : petal_l,
                    "Petal W"   : petal_w,
                    "Prediction": species,
                    "Confidence": f"{confidence:.1f}%",
                })
                st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

    with col_res:
        if "last_prediction" in st.session_state:
            p = st.session_state.last_prediction
            display_result(p["species"], p["confidence"], p["all_probs"])
        else:
            st.markdown(
                """
                <div class="glass-card" style="text-align:center;padding:3rem 1.5rem;opacity:.65;">
                  <div style="font-size:3.5rem;margin-bottom:1rem;">🌸</div>
                  <div style="font-weight:700;color:#1a3d26;font-size:1.05rem;margin-bottom:.4rem;">
                    Awaiting Prediction
                  </div>
                  <div style="font-size:.85rem;color:#6b8f77;">
                    Adjust the sliders on the left and click<br>
                    <strong>Predict Flower Species</strong> to see results here.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Stats ───────────────────────────────────────────────
    render_stats(iris)

    # ── KNN Explainer ───────────────────────────────────────
    render_knn_explainer()

    # ── Model Report ────────────────────────────────────────
    render_model_report(accuracy, cm, report)

    # ── History ─────────────────────────────────────────────
    render_history()

    # ── Footer ──────────────────────────────────────────────
    render_footer(accuracy)


# ── Entry Point ─────────────────────────────────────────────
if __name__ == "__main__":
    main()
