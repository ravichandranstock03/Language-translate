"""
Real-time Voice and Text Translation System

This module provides fast, real-time translation between multiple languages
using pre-trained transformer models. It supports both text and voice input/output.

Features:
- Multi-language support (100+ languages)
- Real-time streaming translation
- Voice-to-voice translation
- Text-to-text translation
- Low latency optimization
"""

import os
import torch
from typing import Optional, Generator, Dict, Any
from dataclasses import dataclass
from enum import Enum


class Language(Enum):
    """Supported languages with their codes"""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    JAPANESE = "ja"
    KOREAN = "ko"
    CHINESE = "zh"
    ARABIC = "ar"
    HINDI = "hi"
    DUTCH = "nl"
    POLISH = "pl"
    TURKISH = "tr"
    VIETNAMESE = "vi"
    THAI = "th"
    INDONESIAN = "id"


@dataclass
class TranslationConfig:
    """Configuration for the translation system"""
    source_language: str = "en"
    target_language: str = "es"
    model_type: str = "nllb"  # Options: nllb, marian, m2m100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_length: int = 512
    beam_size: int = 4
    enable_streaming: bool = True
    cache_model: bool = True


class TextTranslator:
    """
    High-performance text translation using pre-trained models.
    Supports multiple model backends for different use cases.
    """
    
    def __init__(self, config: TranslationConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._model_loaded = False
        
    def load_model(self, model_name: Optional[str] = None) -> None:
        """
        Load the translation model and tokenizer.
        
        Args:
            model_name: Specific model name. If None, uses default based on config
        """
        if self._model_loaded:
            return
            
        if model_name is None:
            if self.config.model_type == "nllb":
                model_name = "facebook/nllb-200-distilled-600M"
            elif self.config.model_type == "marian":
                src = self.config.source_language
                tgt = self.config.target_language
                model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
            elif self.config.model_type == "m2m100":
                model_name = "facebook/m2m100_418M"
            else:
                raise ValueError(f"Unknown model type: {self.config.model_type}")
        
        print(f"Loading model: {model_name}")
        print(f"Using device: {self.config.device}")
        
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.model.to(self.config.device)
            self.model.eval()
            
            if self.config.cache_model:
                # Enable memory optimizations
                if self.config.device == "cuda":
                    self.model.half()  # Use FP16 for faster inference
                    
            self._model_loaded = True
            print("Model loaded successfully!")
            
        except ImportError:
            raise ImportError(
                "Please install transformers: pip install transformers torch sentencepiece"
            )
    
    def translate(self, text: str, stream: bool = False) -> str | Generator[str, None, None]:
        """
        Translate text from source to target language.
        
        Args:
            text: Input text to translate
            stream: If True, returns a generator for streaming output
            
        Returns:
            Translated text or generator yielding chunks
        """
        if not self._model_loaded:
            self.load_model()
        
        if stream:
            return self._stream_translate(text)
        else:
            return self._translate_batch(text)
    
    def _translate_batch(self, text: str) -> str:
        """Batch translation for complete texts"""
        import torch
        
        # Prepare input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length
        ).to(self.config.device)
        
        # Generate translation
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_length=self.config.max_length,
                num_beams=self.config.beam_size,
                early_stopping=True
            )
        
        # Decode output
        translation = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        return translation
    
    def _stream_translate(self, text: str) -> Generator[str, None, None]:
        """Streaming translation for real-time output"""
        # For true streaming, we'd process chunks of text
        # This is a simplified version that yields the result
        result = self._translate_batch(text)
        yield result
    
    def translate_batch(self, texts: list[str]) -> list[str]:
        """Translate multiple texts efficiently"""
        if not self._model_loaded:
            self.load_model()
        
        import torch
        
        # Tokenize all texts
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_length
        ).to(self.config.device)
        
        # Generate translations
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_length=self.config.max_length,
                num_beams=self.config.beam_size,
                early_stopping=True
            )
        
        # Decode all outputs
        translations = [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in generated_ids
        ]
        
        return translations


class VoiceTranslator:
    """
    Voice translation system combining speech-to-text, translation, and text-to-speech.
    """
    
    def __init__(self, config: TranslationConfig):
        self.config = config
        self.text_translator = TextTranslator(config)
        self.stt_model = None  # Speech-to-text model
        self.tts_model = None  # Text-to-speech model
        self._stt_loaded = False
        self._tts_loaded = False
    
    def load_speech_to_text(self, model_name: str = "base") -> None:
        """
        Load speech-to-text model (Whisper).
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
        """
        if self._stt_loaded:
            return
        
        print(f"Loading Whisper model: {model_name}")
        
        try:
            import whisper
            
            self.stt_model = whisper.load_model(model_name, device=self.config.device)
            self._stt_loaded = True
            print("Speech-to-text model loaded!")
            
        except ImportError:
            raise ImportError(
                "Please install openai-whisper: pip install openai-whisper"
            )
    
    def load_text_to_speech(self, provider: str = "google") -> None:
        """
        Load text-to-speech model.
        
        Args:
            provider: TTS provider (google, pyttsx3, coqui)
        """
        if self._tts_loaded:
            return
        
        print(f"Loading TTS engine: {provider}")
        
        try:
            if provider == "pyttsx3":
                import pyttsx3
                self.tts_model = pyttsx3.init()
                self.tts_model.setProperty('rate', 150)
            elif provider == "google":
                from gtts import gTTS
                self.tts_model = gTTS
            elif provider == "coqui":
                from TTS.api import TTS
                self.tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False)
            else:
                raise ValueError(f"Unknown TTS provider: {provider}")
                
            self._tts_loaded = True
            print("Text-to-speech engine loaded!")
            
        except ImportError as e:
            raise ImportError(
                f"Please install required TTS library: {e}\n"
                "Options: pip install pyttsx3 OR gTTS OR TTS"
            )
    
    def speech_to_text(self, audio_path: str) -> str:
        """
        Convert speech audio file to text.
        
        Args:
            audio_path: Path to audio file (wav, mp3, etc.)
            
        Returns:
            Transcribed text
        """
        if not self._stt_loaded:
            self.load_speech_to_text()
        
        result = self.stt_model.transcribe(
            audio_path,
            language=self.config.source_language
        )
        
        return result["text"]
    
    def text_to_speech(self, text: str, output_path: str = "output.mp3") -> str:
        """
        Convert translated text to speech.
        
        Args:
            text: Text to synthesize
            output_path: Path to save audio file
            
        Returns:
            Path to generated audio file
        """
        if not self._tts_loaded:
            self.load_text_to_speech()
        
        if hasattr(self.tts_model, 'save'):
            # pyttsx3
            self.tts_model.save_to_file(text, output_path)
            self.tts_model.runAndWait()
        elif self.tts_model == gTTS:
            # Google TTS
            tts = gTTS(text=text, lang=self.config.target_language)
            tts.save(output_path)
        else:
            # Coqui TTS
            self.tts_model.tts_to_file(text=text, file_path=output_path)
        
        return output_path
    
    def translate_voice(self, audio_path: str, output_audio_path: str = "translated.mp3") -> Dict[str, Any]:
        """
        Complete voice-to-voice translation pipeline.
        
        Args:
            audio_path: Input audio file path
            output_audio_path: Output translated audio file path
            
        Returns:
            Dictionary with original text, translated text, and output audio path
        """
        # Step 1: Speech to Text
        print("Converting speech to text...")
        original_text = self.speech_to_text(audio_path)
        print(f"Original: {original_text}")
        
        # Step 2: Translate Text
        print("Translating text...")
        translated_text = self.text_translator.translate(original_text)
        print(f"Translated: {translated_text}")
        
        # Step 3: Text to Speech
        print("Converting text to speech...")
        output_path = self.text_to_speech(translated_text, output_audio_path)
        
        return {
            "original_text": original_text,
            "translated_text": translated_text,
            "output_audio": output_path
        }


class RealTimeTranslator:
    """
    Real-time streaming translator for live voice/text translation.
    Optimized for low-latency applications.
    """
    
    def __init__(self, config: TranslationConfig):
        self.config = config
        self.voice_translator = VoiceTranslator(config)
        self.is_running = False
    
    async def translate_audio_stream(self, audio_stream, chunk_size: int = 1024) -> Generator[Dict, None, None]:
        """
        Translate audio in real-time from a stream.
        
        Args:
            audio_stream: Audio stream iterator
            chunk_size: Size of audio chunks to process
            
        Yields:
            Dictionary with partial translations
        """
        self.is_running = True
        
        # Note: Full real-time implementation would require:
        # - VAD (Voice Activity Detection)
        # - Streaming Whisper model
        # - Incremental translation
        # - Streaming TTS
        
        # This is a simplified version
        audio_buffer = []
        
        async for chunk in audio_stream:
            if not self.is_running:
                break
                
            audio_buffer.append(chunk)
            
            # Process when we have enough audio
            if len(audio_buffer) >= 10:  # Adjust threshold as needed
                # Combine chunks
                import numpy as np
                combined = np.concatenate(audio_buffer)
                
                # Save temporary file
                import tempfile
                import soundfile as sf
                
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    sf.write(tmp.name, combined, 16000)
                    temp_path = tmp.name
                
                # Translate
                result = self.voice_translator.translate_voice(temp_path)
                
                yield {
                    "status": "partial",
                    **result
                }
                
                # Clear buffer
                audio_buffer = []
                os.unlink(temp_path)
        
        # Process remaining audio
        if audio_buffer:
            import numpy as np
            combined = np.concatenate(audio_buffer)
            
            import tempfile
            import soundfile as sf
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                sf.write(tmp.name, combined, 16000)
                temp_path = tmp.name
            
            result = self.voice_translator.translate_voice(temp_path)
            result["status"] = "complete"
            
            yield result
            os.unlink(temp_path)
        
        self.is_running = False
    
    def stop(self):
        """Stop real-time translation"""
        self.is_running = False


def create_translator(
    source_lang: str = "en",
    target_lang: str = "es",
    mode: str = "text"
) -> TextTranslator | VoiceTranslator | RealTimeTranslator:
    """
    Factory function to create appropriate translator.
    
    Args:
        source_lang: Source language code
        target_lang: Target language code
        mode: Translation mode (text, voice, realtime)
        
    Returns:
        Configured translator instance
    """
    config = TranslationConfig(
        source_language=source_lang,
        target_language=target_lang
    )
    
    if mode == "text":
        return TextTranslator(config)
    elif mode == "voice":
        return VoiceTranslator(config)
    elif mode == "realtime":
        return RealTimeTranslator(config)
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose from: text, voice, realtime")


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Real-time Translation System Demo")
    print("=" * 60)
    
    # Text Translation Example
    print("\n1. Text Translation:")
    translator = create_translator("en", "es", mode="text")
    text = "Hello, how are you today?"
    result = translator.translate(text)
    print(f"   English: {text}")
    print(f"   Spanish: {result}")
    
    # Voice Translation Example (requires audio file)
    print("\n2. Voice Translation:")
    print("   To test voice translation, provide an audio file:")
    print("   translator = create_translator('en', 'fr', mode='voice')")
    print("   result = translator.translate_voice('input.wav', 'output.mp3')")
    
    print("\n" + "=" * 60)
    print("Installation Requirements:")
    print("=" * 60)
    print("""
    pip install transformers torch sentencepiece
    pip install openai-whisper
    pip install gTTS pyttsx3
    pip install soundfile numpy
    
    For GPU acceleration:
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    """)
