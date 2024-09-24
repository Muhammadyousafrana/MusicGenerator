import streamlit as st
from transformers import AutoTokenizer, AutoModelForTextToWaveform
import torch
import soundfile as sf
import numpy as np

# Streamlit app title
st.title("Text to Music Generation with MusicGen")

@st.cache_resource(show_spinner=True)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("facebook/musicgen-small")
    model = AutoModelForTextToWaveform.from_pretrained("facebook/musicgen-small")
    return tokenizer, model

# Input text area for the user to enter a prompt
user_input = st.text_area("Enter a prompt for music generation:")

if st.button("Generate Music"):
    if user_input:
        # Load the model when needed
        with st.spinner("Loading model..."):
            tokenizer, model = load_model()

        # Tokenize the input
        input_ids = tokenizer(user_input, return_tensors="pt").input_ids

        # Generate audio waveform
        with torch.no_grad():
            waveform = model.generate(input_ids, num_return_sequences=1)

        # Convert waveform to numpy and save as .wav
        waveform_np = waveform[0].cpu().numpy()
        sf.write("output.wav", waveform_np, 44100)

        # Display audio player
        st.audio("output.wav", format='audio/wav')
    else:
        st.warning("Please enter a prompt before generating music.")
