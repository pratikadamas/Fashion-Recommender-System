import streamlit as st
import numpy as np
import pickle
from PIL import Image
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

# -----------------------
# Streamlit config
# -----------------------
st.set_page_config(layout="wide")

# -----------------------
# Session state
# -----------------------
if "selected_image" not in st.session_state:
    st.session_state.selected_image = None

# -----------------------
# Load model
# -----------------------
@st.cache_resource
def load_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        GlobalMaxPooling2D()
    ])
    return model

model = load_model()

# -----------------------
# Load data
# -----------------------
@st.cache_data
def load_data():
    feature_list = pickle.load(open('embeddings.pkl','rb'))
    filenames = pickle.load(open('filenames.pkl','rb'))
    return np.array(feature_list), filenames

feature_list, filenames = load_data()

# -----------------------
# Feature extraction
# -----------------------
def extract_features(img):
    img = img.resize((224,224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)

    result = model.predict(preprocessed_img, verbose=0).flatten()
    normalized_result = result / norm(result)

    return normalized_result

# -----------------------
# Recommendation
# -----------------------
def recommend(features, feature_list):
    if feature_list.shape[0] == 0:
        raise Exception("Feature list is empty. Run extract_features.py first.")

    similarities = cosine_similarity(
        features.reshape(1, -1),
        feature_list
    )[0]

    indices = np.argsort(similarities)[::-1][1:6]
    return indices

# -----------------------
# UI
# -----------------------
st.title("👕 Fashion Recommender System")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)

    st.image(img, caption="Uploaded Image", width=300)

    features = extract_features(img)

    try:
        indices = recommend(features, feature_list)

        st.subheader("Top 5 Similar Products")

        cols = st.columns(5)

        for i, col in enumerate(cols):
            with col:
                st.image(filenames[indices[i]])

                if st.button(f"🔍 View", key=f"btn_{i}"):
                    st.session_state.selected_image = filenames[indices[i]]

    except Exception as e:
        st.error(str(e))

# -----------------------
# Enlarged image view
# -----------------------
if st.session_state.selected_image is not None:
    st.markdown("---")
    st.subheader("🔍 Image Preview")

    st.image(st.session_state.selected_image, width=500)

    if st.button("❌ Close Preview"):
        st.session_state.selected_image = None