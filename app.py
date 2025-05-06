import streamlit as st
import numpy as np
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from transformers import pipeline

# Set Streamlit page config
st.set_page_config(
    page_title="🌐 Language Detection App",
    page_icon="🌍",
    layout="centered"
)

st.sidebar.title("🌍 Supported Languages")
st.sidebar.markdown("""
The models can detect the following **Indian languages**:

- 🅿️ Punjabi  
- 🅾️ Odia  
- 🕉️ Konkani  
- 🕊️ Hindi  
- 📜 Sanskrit  
- 🧁 Bengali  
- 🇬🇧 English  
- 🧕 Sindhi  
- 🏔️ Nepali  
- 🐅 Marathi  
- 🪔 Gujarati  
- 🎶 Telugu  
- 🌴 Malayalam  
- 🎨 Tamil  
- 🐘 Kannada  
- ❄️ Kashmiri  
- ☪️ Urdu  
- 🌾 Assamese  
""")


# ---------- Header ----------
st.markdown("<h1 style='text-align: center; color: #2c3e50;'>🌍🇮🇳 Indian Language Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter a sentence in any Indian language, and get predictions from both my custom-trained model and a DistilBERT-based feature-extraction model for language detection.</p>", unsafe_allow_html=True)

st.markdown("---")

# ---------- Cache Models ----------
@st.cache_resource
def load_custom_model():
    model = load_model("language_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return model, tokenizer, le

@st.cache_resource
def load_transformer_model():
    classifier = pipeline("text-classification", 
                      model="Bvadaliya3005/language-detector",
                      tokenizer="Bvadaliya3005/language-detector")
    return classifier

custom_model, tokenizer, le = load_custom_model()
transformer = load_transformer_model()

# ---------- Input ----------
user_input = st.text_area("✍️ Enter your sentence here:", height=150, placeholder="Type something...")

if st.button('🔍 Predict'):
    if user_input:
        st.markdown("## 🧠 Model Predictions")

        # ---------- Custom Model Prediction ----------
        seq = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(seq, padding='post', maxlen=1000)
        pred = custom_model.predict(padded)
        custom_pred_label = le.inverse_transform([np.argmax(pred)])[0]

        # ---------- DistilBERT Prediction ----------
        result = transformer(user_input)[0]
        label_index = int(result['label'].split("_")[1])
        transformer_label = le.inverse_transform([label_index])[0]

        # ---------- Results ----------
        col1, col2 = st.columns(2)
        with col1:
            st.success("🧬 **Custom Model Prediction**")
            st.markdown(f"🗣️ **Language**: `{custom_pred_label}`")

        with col2:
            st.info("🤖 **DistilBERT Prediction**")
            st.markdown(f"🗣️ **Language**: `{transformer_label}`")
            st.markdown(f"📈 **Confidence**: `{result['score']:.4f}`")
    else:
        st.warning("⚠️ Please enter some text in an Indian language.")


st.markdown("---")

# ---------- Footer ----------
st.markdown("<p style='text-align: center; font-size: 0.8em; color: gray;'>Built with ❤️ using Keras, Hugging Face, and Streamlit</p>", unsafe_allow_html=True)
