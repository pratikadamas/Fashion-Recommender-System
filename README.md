# 👗 Fashion Recommender System

A deep learning–based image similarity system that recommends visually similar fashion products using feature embeddings extracted from a pretrained CNN model.

---

## 🚀 Overview

This project uses a pretrained **ResNet50** model to extract high-dimensional feature vectors from images. These embeddings are then compared using cosine similarity to recommend visually similar items.

👉 Instead of classification, this system performs **image similarity search**.

---

## 🧠 How It Works

1. Input image is resized to **224×224**
2. Passed through pretrained **ResNet50 (without top layer)**
3. Extracted features are:

   * Flattened using Global Max Pooling
   * Normalized into unit vectors
4. Similarity is computed using **cosine similarity**
5. Top-N closest images are returned as recommendations

---

## 🏗️ Tech Stack

* Python
* TensorFlow / Keras
* NumPy
* Scikit-learn
* Streamlit (for UI)

---

## 📂 Project Structure

```
fashion-recommender/
│
├── app.py                  # Streamlit app
├── extract_features.ipynb  # Feature extraction notebook
├── embeddings.pkl          # Stored feature vectors (ignored in Git)
├── filenames.pkl           # Image paths (ignored in Git)
├── images/                 # Dataset (ignored in Git)
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/Fashion-Recommender-System.git
cd Fashion-Recommender-System

pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## 📊 Feature Extraction

To generate embeddings:

```bash
python extract_features.py
```

This will create:

* `embeddings.pkl`
* `filenames.pkl`

---

## ⚠️ Important Notes

* Large files like datasets and `.pkl` files are excluded using `.gitignore`
* Model uses **ImageNet pretrained weights**
* Ensure consistent preprocessing using `preprocess_input`

---

## 📈 Future Improvements

* Use FAISS for faster similarity search
* Add filtering (category, price, brand)
* Deploy on cloud (Streamlit Cloud / Render)
* Improve UI/UX

---

## 💡 Key Concept

This project is based on:

```
Image → CNN → Feature Vector → Normalize → Compare → Recommend
```

---

## 👤 Author

**Pratik Giri**
Computer Science Student | AI/ML Enthusiast

---

## 📜 License

This project is for educational purposes.
![alt text](<Screenshot 2026-04-29 233648.png>)
![alt text](<Screenshot 2026-04-29 233707.png>)
