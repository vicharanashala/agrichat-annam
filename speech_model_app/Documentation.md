#  Language-Aware Transcription App

This is a Gradio-based web interface for **automatic language detection and transcription** of audio files using two powerful models:
- [`openai/whisper-large`](https://huggingface.co/openai/whisper-large) for language detection
- [`ai4bharat/indic-conformer-600m-multilingual`](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual) for multilingual speech transcription using both CTC and RNNT decoders.

---

##  Features

- **Automatic Language Detection** with Whisper
- **Dual Transcription Output** using CTC and RNNT decoders
- Supports **WAV and MP3** audio files
- Lightweight and easy-to-deploy using **Gradio**

---

##  Demo

The app provides a simple web UI:

1. Upload an audio file (WAV/MP3).
2. The app detects the language.
3. The transcription is generated using both CTC and RNNT decoders.

You can try the live demo here:  
👉 [Hugging Face Space Demo](https://huggingface.co/spaces/Noumida/Speech_to_Text_LID_Transcription_2)

---

##  Files

- `app.py`: Main Gradio app for language detection and transcription
- `Documentation.md`
- `Requirements.txt`
- `.gitattributes`

---

##  Credits

- [OpenAI Whisper]
- [AI4Bharat Indic Conformer]
- [Gradio]
- [Hugging Face]

---

##  License

This project is licensed under the MIT License.
