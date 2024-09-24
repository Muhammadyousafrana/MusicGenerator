import streamlit as st
from transformers import AutoTokenizer, AutoModelForTextToWaveform
import torch
import soundfile as sf
import numpy as np

# Streamlit app title with CSS styling
st.markdown("""
    <style>
    body {
        background-color: #f0f2f6;
    }
    .main-title {
        font-size: 40px;
        color: #4CAF50;
        text-align: center;
        font-family: 'Arial', sans-serif;
    }
    .text-input {
        margin: 20px 0;
        font-size: 18px;
    }
    .btn-generate {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        border: none;
        font-size: 18px;
    }
    .btn-generate:hover {
        background-color: #45a049;
    }
    .stAudio {
        margin-top: 20px;
    }
    .warning-text {
        color: #FF6F61;
    }
    </style>
""", unsafe_allow_html=True)

# Main app title
st.markdown('<h1 class="main-title">🎶 Text to Music Generation with MusicGen 🎶</h1>', unsafe_allow_html=True)

# Cache the model and tokenizer
@st.cache_resource(show_spinner=True)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("facebook/musicgen-small")
    model = AutoModelForTextToWaveform.from_pretrained("facebook/musicgen-small")
    return tokenizer, model

# Cache the generation process
@st.cache_data(show_spinner=True)
def generate_music(prompt, tokenizer, model):
    # Tokenize the input
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    
    # Generate audio waveform
    with torch.no_grad():
        waveform = model.generate(input_ids, num_return_sequences=1)
    
    # Convert waveform to numpy for saving
    waveform_np = waveform[0].cpu().numpy()
    return waveform_np

# Input text area for the user to enter a prompt
st.markdown('<p class="text-input">Enter a prompt for music generation:</p>', unsafe_allow_html=True)
user_input = st.text_area("", height=120, key="text_area_input")

# Generate button with custom CSS styling
if st.markdown('<button class="btn-generate">Generate Music</button>', unsafe_allow_html=True):
    if user_input:
        # Load the model and tokenizer
        with st.spinner("Loading model..."):
            tokenizer, model = load_model()

        # Generate music
        with st.spinner("Generating music..."):
            waveform_np = generate_music(user_input, tokenizer, model)

        # Save waveform as .wav file
        sf.write("output.wav", waveform_np, 44100)

        # Display audio player
        st.audio("output.wav", format='audio/wav')
    else:
        st.markdown('<p class="warning-text">⚠️ Please enter a prompt before generating music.</p>', unsafe_allow_html=True)
