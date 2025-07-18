import gradio as gr
import torchaudio
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# Whisper model for language detection
lang_id_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium").to(device)
lang_id_processor = WhisperProcessor.from_pretrained("openai/whisper-medium")

# Indic Conformer model for transcription
model = AutoModel.from_pretrained("ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True)

def detect_language(audio_path):
    waveform, sr = torchaudio.load(audio_path)
    waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
    inputs = lang_id_processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").to(device)

    start_token_id = lang_id_processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    decoder_input_ids = torch.tensor([[start_token_id]], device=device)

    with torch.no_grad():
        outputs = lang_id_model.generate(
            inputs["input_features"],
            decoder_input_ids=decoder_input_ids,
            max_new_tokens=1,
        )

    lang_token = lang_id_processor.tokenizer.decode(outputs[0], skip_special_tokens=False)
    lang_code = lang_token.replace("<|", "").replace("|>", "").strip()
    return lang_code, waveform.to(device)

def transcribe(audio_path):
    try:
        lang_code, wav = detect_language(audio_path)

        transcription_ctc = model(wav, lang_code, "ctc")
        transcription_rnnt = model(wav, lang_code, "rnnt")

        return lang_code, transcription_ctc, transcription_rnnt
    except Exception as e:
        return "Error", str(e), ""

demo = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath", label="Upload a WAV or MP3 file"),
    outputs=[
        gr.Textbox(label="Detected Language"),
        gr.Textbox(label="CTC Transcription"),
        gr.Textbox(label="RNNT Transcription")
    ],
    title="Language-Aware Transcription",
    description="Step 1: Detect language using Whisper. Step 2: Transcribe using AI4Bharat Indic Conformer."
)

if __name__ == "__main__":
    demo.launch()

