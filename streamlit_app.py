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
    
tokenizer = pickle.load(open('tokenizer (1).pkl', 'rb'))
model_lstm = load_model("lstm_model.h5")
model_bilstm = load_model("bilstm_model (1).h5")
model_rnn = load_model("rnn_model (1).h5")

MAX_LEN = 512

# Define your label mapping manually
bias_labels = ['left', 'right', 'center']

def predict_lstm(text):
    try:
        sequence = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')

        prediction = model_lstm.predict(padded)
        predicted_index = prediction.argmax(axis=1)[0]
        predicted_label = bias_labels[predicted_index]

        return predicted_label

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return "Prediction Failed"
        
def predict_bilstm(text):
     try:
        sequence = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')

        prediction = model_bilstm.predict(padded)
        predicted_index = prediction.argmax(axis=1)[0]
        predicted_label = bias_labels[predicted_index]

        return predicted_label

     except Exception as e:
         st.error(f"Prediction error: {str(e)}")
         return "Prediction Failed"

def predict_rnn(text):
     try:
        sequence = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')

        prediction = model_rnn.predict(padded)
        predicted_index = prediction.argmax(axis=1)[0]
        predicted_label = bias_labels[predicted_index]

        return predicted_label

     except Exception as e:
         st.error(f"Prediction error: {str(e)}")
         return "Prediction Failed"
    
st.title(" Bias Detection Using Pre-trained Model")
text = st.text_area("Enter a news article :")

model_choice = st.selectbox("Choose a model for bias detection", ["BiLSTM", "LSTM","RNN"])

if st.button("Detect Bias") and text:
    if model_choice == "BiLSTM":
        predicted_bias = predict_bilstm(text)
    elif model_choice == "LSTM":
        predicted_bias = predict_lstm(text)
    elif model_choice == "RNN":
        predicted_bias = predict_rnn(text)
    else:
        predicted_bias = "Unknown model selected"

    st.markdown(f"🧠 **Predicted Bias:** `{predicted_bias}`")
    if predicted_bias == "left":
        st.markdown("👈This article seems **left-leaning**.")
    elif predicted_bias == "right":
        st.markdown("👉 This article seems **right-leaning**.")
    else:
        st.markdown("✋ This article appears **center**.")

    
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]


GROQ_HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json"
}


def summarize_with_groq(text, model="meta-llama/llama-4-scout-17b-16e-instruct"):
    try:
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": f"Summarize the article:\n\n{text}"}
            ],
            "temperature": 0.7,
            "max_tokens": 512
        }

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=GROQ_HEADERS,
            json=payload,
            timeout=60
        )
        data = response.json()
        return data["choices"][0]["message"]["content"]
    
    except Exception as e:
        return f"Error: {e}"

# Streamlit UI
st.title("Article Summarizer")
model_choice = st.selectbox("Choose Model", ["Gemini", "Groq - LLaMA", " Mistral"])
user_article = st.text_area("Your Text Here...")

if user_article:
    if model_choice == "Gemini ":
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"summarize the article:\n\n{user_article}"
        )
        

    elif model_choice == "Groq - LLaMA":
        summary = summarize_with_groq(user_article, model="meta-llama/llama-4-scout-17b-16e-instruct")
       


    with st.container():
        st.write("summarized article:")
        st.write(response.text)
