# 🌍🇮🇳 Indian Language Detection App

A simple and interactive Streamlit web app that detects which Indian language a sentence is written in — powered by a custom deep Bidirectional LSTM model and a DistilBERT-based transformer model (feature extraction mode).

This tool supports 18 Indian languages and offers predictions from both models side-by-side.

---

# 🧠 Supported Languages

The models are trained to recognize the following Indian languages:

- Punjabi, Odia, Konkani, Hindi, Sanskrit, Bengali, English, Sindhi, Nepali, Marathi, Gujarati, Telugu, Malayalam, Tamil, Kannada, Kashmiri, Urdu, Assamese

---

# 🚀 How to Run
## 1. Install Required Packages:
```bash
  pip install streamlit tensorflow scikit-learn transformers
```
## 2. Launch the Streamlit App:
```bash
  streamlit run app.py
```

---

# 🧠 Model Summary
## 🔷 Custom Model – Bidirectional LSTM
- Built using TensorFlow/Keras
- Deep Bidirectional LSTM architecture for context from both directions
- Uses Tokenizer and padded sequences
- Outputs class index → mapped to actual language via LabelEncoder
- Designed for sentence-level prediction of Indian languages

## 🤖 DistilBERT Model (Feature Extractor)
- Pre-trained DistilBERT model used as a frozen feature extractor
- Predictions decoded using LabelEncoder
- Returns predicted language and confidence score

---

# ✍️ Example Sentences

| Language     | Sentence                                                     |
| ------------ | ------------------------------------------------------------ |
| **Hindi**    | कृत्रिम बुद्धिमत्ता आधुनिक तकनीक का एक प्रमुख उदाहरण है।     |
| **Punjabi**  | ਕ੍ਰਿਤਰਿਮ ਬੁੱਧੀ ਮਸ਼ੀਨਾਂ ਨੂੰ ਮਨੁੱਖਾਂ ਵਾਂਗ ਸੋਚਣ ਯੋਗ ਬਣਾਉਂਦੀ ਹੈ। |
| **Gujarati** | કૃત્રિમ બુદ્ધિ ભવિષ્યની ટેકનોલોજીનું મહત્વપૂર્ણ અંગ છે.      |
| **Konkani**  | कृत्रिम बुद्धिमत्ता माणसाच्या बुद्धीप्रमाणे विचार करू शकते.  |
| **Urdu**     | مصنوعی ذہانت جدید ٹیکنالوجی کا ایک حیرت انگیز میدان ہے۔      |





