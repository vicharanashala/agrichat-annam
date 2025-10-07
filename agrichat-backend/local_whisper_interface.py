"""
Local Whisper Interface for Speech-to-Text
Replaces Hugging Face API with local GPU inference
"""

import whisper
import tempfile
import os
from typing import Optional

class LocalWhisperInterface:
    """Interface for local Whisper model"""
    
    def __init__(self, model_size: str = "base"):
        """
        Initialize Whisper model
        
        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
                       - tiny: fastest, least accurate
                       - base: good balance of speed/accuracy  
                       - small: better accuracy, slower
                       - medium: very good accuracy
                       - large: best accuracy, slowest
        """
        self.model_size = model_size
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load Whisper model from cache directory"""
        try:
            # Set the cache directory for Whisper
            cache_dir = os.environ.get('WHISPER_CACHE_DIR', os.path.expanduser('~/.cache/whisper'))
            print(f"📥 Loading Whisper {self.model_size} model from cache...")
            print(f"💾 Cache directory: {cache_dir}")
            
            # Load model directly without checking cache (model should already be cached)
            self.model = whisper.load_model(self.model_size, download_root=cache_dir)
            print(f"Whisper {self.model_size} model loaded successfully from cache")
            
            import torch
            if torch.cuda.is_available():
                print(f"Using GPU acceleration with {torch.cuda.get_device_name()}")
            else:
                print("💻 Using CPU inference (GPU not available)")
                
        except Exception as e:
            print(f"Failed to load Whisper model: {e}")
            self.model = None
    
    def transcribe_audio(self, audio_data: bytes, filename: str = "audio") -> str:
        """
        Transcribe audio from bytes
        
        Args:
            audio_data: Audio file bytes
            filename: Original filename for debugging
            
        Returns:
            Transcribed text
        """
        if not self.model:
            return "Error: Whisper model not loaded"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=self._get_audio_extension(filename)) as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name
        
        try:
            print(f"🎤 Transcribing audio: {filename}")
            result = self.model.transcribe(temp_file_path)
            transcript = result["text"].strip()
            
            print(f"Transcription successful: {transcript[:100]}...")
            return transcript
            
        except Exception as e:
            print(f"Transcription failed: {e}")
            return f"Error: Failed to transcribe audio - {str(e)}"
            
        finally:
            try:
                os.unlink(temp_file_path)
            except:
                pass
    
    def _get_audio_extension(self, filename: str) -> str:
        """Get appropriate file extension"""
        if not filename:
            return ".wav"
        
        ext = os.path.splitext(filename)[1].lower()
        if ext in ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.webm']:
            return ext
        return ".wav"

# Global Whisper instance - loaded immediately when module is imported
print("[WHISPER] Initializing Whisper interface with base model...")
_whisper_instance = LocalWhisperInterface(model_size="base")

def get_whisper_instance(model_size: str = "base") -> LocalWhisperInterface:
    """
    Get the global Whisper instance (pre-loaded, no lazy loading)
    
    Args:
        model_size: Whisper model size (ignored, always uses pre-loaded base model)
    
    Returns:
        LocalWhisperInterface instance (pre-loaded)
    """
    # Return the pre-loaded instance regardless of model_size parameter
    return _whisper_instance


