import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
import re
import random
import nltk
from nltk.corpus import stopwords
import spacy
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from rouge_score import rouge_scorer
from bert_score import score as bertscore
import requests
import google.generativeai as genai
from google import genai

st.title('Media Bias Detection and Summarization App')

st.write('This app is for Bias detection and Summarization of News Articles !')
st.sidebar.title('Media Bias Detection and Summarization')

st.title("Hugging Face Dataset Viewer")

@st.cache_data
def load_data():
    dataset = load_dataset("Faith1712/Allsides_political_bias_proper", split="train[:15000]")
    return dataset
dataset = load_data()
df = pd.DataFrame(dataset)
with st.expander('Analysis Data'):
    st.write("Raw Data")
    st.dataframe(df)
    
tokenizer = pickle.load(open('tokenizer (1).pkl', 'rb'))
model = load_model("lstm_model.h5")

MAX_LEN = 512

# Define your label mapping manually
bias_labels = ['left', 'right', 'center']

def predict_bias(text):
    try:
        sequence = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')

        prediction = model.predict(padded)
        predicted_index = prediction.argmax(axis=1)[0]
        predicted_label = bias_labels[predicted_index]

        return predicted_label

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return "Prediction Failed"

    
st.title(" Bias Detection Using Pre-trained Model")
text = st.text_area("Enter a news article :")

if st.button("Detect Bias") and text:
    predicted_bias = predict_bias(text)

    st.markdown(f"🧠 **Predicted Bias:** `{predicted_bias}`")
    if predicted_bias == "left":
        st.markdown("👈This article seems **left-leaning**.")
    elif predicted_bias == "right":
        st.markdown("👉 This article seems **right-leaning**.")
    else:
        st.markdown("✋ This article appears **center**.")


st.title("Article Summarizer")
user_article= st.text_area("Your Text Here...")         
if user_article:
    client = genai.Client(api_key= st.secrets[""])
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=f"summarize the article:\n\n{user_article}"
    )
    print(response.text)
    with st.container():
        st.write("summarized article:")
        st.write(response.text)
