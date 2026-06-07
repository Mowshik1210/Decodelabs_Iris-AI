<div align="center">

<!-- BANNER -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=🌸%20Iris%20Flower%20Classifier&fontSize=40&fontColor=fff&animation=fadeIn&fontAlignY=38&desc=AI-Powered%20Species%20Prediction%20using%20K-Nearest%20Neighbors&descAlignY=60&descAlign=50" width="100%"/>

<!-- BADGES -->
[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-f7931e?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)


<br/>

> **🏫 Project · IRIS AI**
> 
> ** 🌐 LIVE DEMO:![🌸 IRISFLOWERAI.STREAMLIT.APP](https://irisflowerai.streamlit.app/)  **
> A production-quality machine learning web application that classifies Iris flower species
> from physical measurements using the K-Nearest Neighbors algorithm —
> wrapped in a premium botanical-themed Streamlit UI.

<br/>

[![⭐ Star this repo](https://img.shields.io/github/stars/Mowshik1210?style=social)](https://github.com/Mowshik1210)
&nbsp;&nbsp;
[![🍴 Fork](https://img.shields.io/github/forks/Mowshik1210?style=social)](https://github.com/Mowshik1210)

</div>

---

## 📸 App Preview

<div align="center">

| Hero Banner | Prediction Result | Species Cards |
|:-----------:|:-----------------:|:-------------:|
| *![](https://github.com/Mowshik1210/Decodelabs_Iris-AI/blob/main/Assests/Screenshot%20(100).png?raw=true)* | *![](https://github.com/Mowshik1210/Decodelabs_Iris-AI/blob/main/Assests/Screenshot%20(102).png?raw=true)* | *![](https://github.com/Mowshik1210/Decodelabs_Iris-AI/blob/main/Assests/Screenshot%20(103).png?raw=true)* |

> 💡 **Tip:** Run the app locally and take screenshots to fill these in!

</div>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🌿 **Botanical UI** | Premium glassmorphism design with `Playfair Display` typography |
| 🎛️ **Live Sliders** | 4 real-time measurement sliders with live cm value badges |
| 🔬 **KNN Prediction** | Scikit-Learn KNN (k=5) with StandardScaler normalisation |
| 📊 **Confidence Score** | Animated confidence bar + full probability distribution |
| 🖼️ **Species Images** | Flower photo display with graceful emoji fallback |
| 📋 **Prediction History** | Session-state table of all predictions in the current session |
| ⬇️ **CSV Export** | Download your entire prediction history as a `.csv` report |
| 🧠 **KNN Explainer** | 5-step visual explainer card on how the algorithm works |
| 📈 **Model Report** | Confusion matrix + full classification report on demand |
| 📊 **Dataset Stats** | Feature statistics and per-class sample breakdown |

---

## 🌸 The Iris Dataset

The **Iris dataset**, introduced by statistician Ronald Fisher in 1936, is one of the most celebrated benchmarks in machine learning.

```
📦 150 total samples  ·  3 species classes  ·  50 samples per class  ·  4 features
```

### 🌷 Species

| Species | Emoji | Origin | Key Trait |
|---------|-------|--------|-----------|
| *Iris setosa* | 🌷 | Arctic & Subarctic regions | Smallest petals · Linearly separable |
| *Iris versicolor* | 🌺 | Eastern North America | Medium size · Blue-violet blooms |
| *Iris virginica* | 💐 | Eastern United States | Largest petals · Coastal wetlands |

### 📐 Features

| Feature | Range | Importance |
|---------|-------|------------|
| Sepal Length (cm) | 4.3 – 7.9 | Moderate |
| Sepal Width (cm) | 2.0 – 4.4 | Lower |
| Petal Length (cm) | 1.0 – 6.9 | **Highest** |
| Petal Width (cm) | 0.1 – 2.5 | **Very High** |

---

## 🤖 Model — K-Nearest Neighbors

```
Algorithm   :  KNeighborsClassifier  (k = 5)
Scaling     :  StandardScaler  (zero mean, unit variance)
Split       :  80% train / 20% test  (random_state = 42)
Accuracy    :  ~97–100%  on test set
```

### How it works in 5 steps

```
① Scale Features  →  ② Compute Euclidean Distance  →  ③ Find 5 Nearest Neighbours
→  ④ Majority Vote  →  ⑤ Return Prediction + Confidence
```

---

## 🗂️ Project Structure

```
iris-flower-classifier/
│
├── 📄 streamlit_app.py       ← Main Streamlit web application
│
├── 📁 Assests/                ← Flower images (add your own!)
│   ├── setosa.jpg
│   ├── versicolor.jpg
│   └── virginica.jpg
│
├── 📄 requirements.txt       ← Python dependencies
└── 📄 README.md              ← You are here
```

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.10+   pip   git
```

### 1 · Clone the repository

```bash
git clone https://github.com/Mowshik1210/Decodelabs_Iris-AI.git
cd iris-flower-classifier
```

### 2 · Install dependencies

```bash
pip install -r requirements.txt
```

### 3 · (Optional) Add flower images

Drop your photos into the `images/` folder:

```bash
mkdir images
# Add setosa.jpg, versicolor.jpg, virginica.jpg
# The app works without images — emoji fallback is built in ✅
```

### 4 · Run the app

```bash
streamlit run streamlit_app.py
```

The app opens automatically at **`http://localhost:8501`**

---

## 📦 Dependencies

```txt
streamlit    >= 1.32.0
scikit-learn >= 1.3.0
pandas       >= 2.0.0
numpy        >= 1.24.0
Pillow       >= 10.0.0
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## 🎯 How to Use

```
1.  Open the app in your browser  (https://irisflowerai.streamlit.app/)
2.  Adjust the 4 measurement sliders  (sepal/petal length & width)
3.  Click  🔬 Predict Flower Species
4.  View the predicted species, confidence %, and probability breakdown
5.  Check prediction history below — download as CSV anytime
```

---

## 📊 Model Performance

```
               precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       1.00      1.00      1.00         9
   virginica       1.00      1.00      1.00        11

    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30
```

> Results may vary slightly depending on your train/test split.

---

## 🧩 Code Architecture

The app is organised into 15 clean, single-responsibility modules:

```python
main()                  # Entry point — wires all sections together
load_model()            # @st.cache_resource — trains & caches KNN + scaler
predict_species()       # Scales inputs → KNN predict → returns name + proba
display_result()        # Renders animated result card + flower image
render_hero()           # Botanical hero banner with floating orbs
render_about()          # Dataset info + 3 species cards
render_feature_cards()  # 4 measurement feature description cards
render_inputs()         # 4 sliders with live value badges
render_stats()          # Dataset statistics + expandable summary
render_knn_explainer()  # 5-step KNN visual explainer
render_model_report()   # Confusion matrix + classification report
render_history()        # Session state prediction log + CSV download
render_footer()         # Project info & credits
```

---

## 🌐 Deploy on Streamlit Cloud *(Free)*

```
1.  Push this repo to GitHub
2.  Go to  →  https://share.streamlit.io
3.  Connect your GitHub account
4.  Select  streamlit_app.py  as the main file
5.  Click  Deploy  —  live in ~2 minutes! 🚀
```

---

## 🛠️ Customisation Guide

**Change k in KNN:**
```python
# In load_model() — line ~90
model = KNeighborsClassifier(n_neighbors=7)   # default is 5
```

**Add your own images:**
```python
# In §1 IMAGE PATHS — top of file
SETOSA_IMAGE     = "images/your_setosa_photo.jpg"
VERSICOLOR_IMAGE = "images/your_versicolor_photo.jpg"
VIRGINICA_IMAGE  = "images/your_virginica_photo.jpg"
```

**Load a pre-saved model instead of training:**
```python
import joblib
model  = joblib.load("model/knn_model.pkl")
scaler = joblib.load("model/scaler.pkl")
```

---

## 📚 Learning Outcomes

By studying this project you will understand:

- ✅ Loading and exploring a real-world ML dataset
- ✅ Feature engineering and StandardScaler normalisation
- ✅ Train / test split strategy and its importance
- ✅ KNN algorithm — theory, hyperparameters, distance metrics
- ✅ Model evaluation — accuracy, confusion matrix, classification report
- ✅ Building interactive ML web apps with Streamlit
- ✅ Session state, caching, and download buttons in Streamlit
- ✅ Professional UI design with custom CSS + glassmorphism

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

```bash
# Fork the repo
git checkout -b feature/your-feature-name
git commit -m "feat: add your feature"
git push origin feature/your-feature-name
# Open a Pull Request
```

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

## 👨‍💻 Author

<div align="center">

**Mowshik G**
*B.E. CSE(AI & ML) *
* KPR institute of engineering and technology *

[![GitHub](https://img.shields.io/badge/GitHub-Mowshik1210-181717?style=for-the-badge&logo=github)](https://github.com/Mowshik1210)

</div>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>

*Built with 💚 for the Earth · Nature-Inspired AI · © 2026 Mowshik G

</div>
