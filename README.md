# MusicGenerator

A Streamlit web app that turns text prompts into short generated music using Hugging Face's MusicGen model.

## Overview

This project provides a simple interface for generating music from natural-language prompts. The app loads the `facebook/musicgen-small` model, creates audio from the entered prompt, and plays the result in the browser.

## Features

- Text prompt input for music generation
- Streamlit-based user interface
- Audio generation with MusicGen
- Browser playback of generated WAV output
- Cached model loading to avoid reloading on every generation

## Project Structure

- `my_app.py` - Main Streamlit application
- `requirements.txt` - Python dependencies
- `README.md` - Project documentation

## Requirements

- Python 3.9+ (recommended: 3.10 or 3.11)
- pip
- A working internet connection to download the model from Hugging Face
- Sufficient RAM and disk space for the model download

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Muhammadyousafrana/MusicGenerator.git
cd MusicGenerator
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Running the App

Start the Streamlit app:

```bash
streamlit run my_app.py
```

Then open the local URL shown in the terminal in your browser.

## Usage

1. Enter a descriptive prompt such as:
   - "Lo-fi beats with soft piano and warm synths"
   - "Upbeat electronic dance track with energetic drums"
   - "Ambient cinematic music with strings and pads"
2. Click the "Generate Music" button.
3. Wait for the model to load and generate audio.
4. Listen to the generated result directly in the browser.

## Notes

- The first run may take several minutes because the model must be downloaded and loaded.
- Generated audio is saved as `output.wav` in the project directory.
- This app is intended for local experimentation and demo use.
- Model performance depends on your hardware, available memory, and internet speed.

## Example Prompt Ideas

- "Calm acoustic guitar melody with gentle rain sounds"
- "Fast-paced electronic music with punchy bass and synth leads"
- "Dreamy orchestral ambience with soft strings"

## Troubleshooting

### Model download fails

- Check your internet connection.
- Ensure the Hugging Face model access is available in your environment.

### Import errors

- Upgrade pip and reinstall requirements:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### App is slow

- Use a machine with more RAM.
- Prefer a GPU-enabled environment if available.
- Keep the model cache warm to avoid repeated downloads.

## License

This project does not currently include a license file. If you plan to distribute or reuse it publicly, consider adding an appropriate open-source license.

## Author

Muhammad Yousaf Rana
