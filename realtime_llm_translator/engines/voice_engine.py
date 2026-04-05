"""
Advanced Voice Processing Engine
Speech-to-Text (Whisper) and Text-to-Speech (VITS/Bark) with streaming support
"""

import torch
import numpy as np
from typing import Optional, Dict, List, Generator, Tuple, Union
from dataclasses import dataclass
import threading
from queue import Queue
import time
import io


@dataclass
class AudioSegment:
    """Container for audio segment data"""
    audio_data: np.ndarray
    sample_rate: int
    duration_ms: float
    language: Optional[str] = None
    is_speech: bool = True


@dataclass
class TranscriptionResult:
    """Container for speech-to-text results"""
    text: str
    language: str
    confidence: float
    latency_ms: float
    segments: List[Dict]
    audio_duration_ms: float


@dataclass
class SpeechGenerationResult:
    """Container for text-to-speech results"""
    audio_data: np.ndarray
    sample_rate: int
    duration_ms: float
    latency_ms: float
    model_used: str


class WhisperBackend:
    """OpenAI Whisper backend for speech-to-text"""
    
    def __init__(
        self,
        model_name: str = "openai/whisper-medium",
        device: str = "auto",
        precision: str = "fp16",
        load_in_8bit: bool = False,
    ):
        self.model_name = model_name
        self.device = self._detect_device(device)
        self.precision = precision
        self.load_in_8bit = load_in_8bit
        self.model = None
        self.processor = None
        self._initialized = False
        self._lock = threading.Lock()
    
    def _detect_device(self, device: str) -> str:
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def initialize(self):
        """Load Whisper model"""
        with self._lock:
            if self._initialized:
                return
            
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            
            print(f"Loading Whisper model: {self.model_name} on {self.device}")
            
            dtype = torch.float16 if self.precision == "fp16" and self.device == "cuda" else torch.float32
            
            self.processor = WhisperProcessor.from_pretrained(self.model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                load_in_8bit=self.load_in_8bit,
                device_map="auto" if self.device == "cuda" else None,
            )
            
            if not self.load_in_8bit and self.device == "cuda":
                self.model.to(self.device)
            
            self.model.eval()
            self._initialized = True
            print("Whisper model loaded successfully")
    
    def transcribe(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
        task: str = "transcribe",
        **kwargs
    ) -> TranscriptionResult:
        """Transcribe audio to text"""
        if not self._initialized:
            self.initialize()
        
        start_time = time.time()
        
        # Ensure audio is mono and correct sample rate
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        
        if sample_rate != 16000:
            # Resample if needed
            import librosa
            audio_data = librosa.resample(
                audio_data.astype(np.float32),
                orig_sr=sample_rate,
                target_sr=16000
            )
            sample_rate = 16000
        
        # Process audio
        input_features = self.processor(
            audio_data,
            sampling_rate=sample_rate,
            return_tensors="pt",
        ).input_features
        
        if self.device == "cuda":
            input_features = input_features.to(self.device)
        
        # Generate transcription
        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features,
                language=language,
                task=task,
                **kwargs
            )
        
        # Decode transcription
        transcription = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]
        
        latency_ms = (time.time() - start_time) * 1000
        audio_duration_ms = (len(audio_data) / sample_rate) * 1000
        
        return TranscriptionResult(
            text=transcription,
            language=language or "en",
            confidence=0.95,
            latency_ms=latency_ms,
            segments=[],
            audio_duration_ms=audio_duration_ms,
        )
    
    def transcribe_streaming(
        self,
        audio_chunks: Generator[np.ndarray, None, None],
        sample_rate: int = 16000,
        language: Optional[str] = None,
        chunk_duration_ms: int = 1000,
    ) -> Generator[str, None, None]:
        """Stream transcription from audio chunks"""
        if not self._initialized:
            self.initialize()
        
        accumulated_audio = []
        
        for chunk in audio_chunks:
            accumulated_audio.append(chunk)
            
            # Process accumulated audio when we have enough
            if sum(len(a) for a in accumulated_audio) >= sample_rate * (chunk_duration_ms / 1000) * 3:
                combined_audio = np.concatenate(accumulated_audio)
                
                result = self.transcribe(
                    combined_audio,
                    sample_rate=sample_rate,
                    language=language,
                )
                
                yield result.text
                accumulated_audio = []
        
        # Process remaining audio
        if accumulated_audio:
            combined_audio = np.concatenate(accumulated_audio)
            result = self.transcribe(
                combined_audio,
                sample_rate=sample_rate,
                language=language,
            )
            yield result.text


class VITSBackend:
    """VITS backend for text-to-speech"""
    
    def __init__(
        self,
        model_name: str = "facebook/mms-tts-eng",
        device: str = "auto",
        precision: str = "fp16",
    ):
        self.model_name = model_name
        self.device = self._detect_device(device)
        self.precision = precision
        self.model = None
        self.processor = None
        self._initialized = False
        self._lock = threading.Lock()
    
    def _detect_device(self, device: str) -> str:
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def initialize(self):
        """Load VITS model"""
        with self._lock:
            if self._initialized:
                return
            
            from transformers import VitsModel, AutoTokenizer
            
            print(f"Loading VITS model: {self.model_name} on {self.device}")
            
            dtype = torch.float16 if self.precision == "fp16" and self.device == "cuda" else torch.float32
            
            self.model = VitsModel.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
            )
            self.processor = AutoTokenizer.from_pretrained(self.model_name)
            
            if self.device == "cuda":
                self.model.to(self.device)
            
            self.model.eval()
            self._initialized = True
            print("VITS model loaded successfully")
    
    def synthesize(
        self,
        text: str,
        language: str = "en",
        speed: float = 1.0,
        **kwargs
    ) -> SpeechGenerationResult:
        """Generate speech from text"""
        if not self._initialized:
            self.initialize()
        
        start_time = time.time()
        
        inputs = self.processor(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, speed=speed)
        
        audio_data = outputs.waveform.cpu().numpy().squeeze()
        
        sample_rate = self.model.config.sampling_rate
        duration_ms = (len(audio_data) / sample_rate) * 1000
        latency_ms = (time.time() - start_time) * 1000
        
        return SpeechGenerationResult(
            audio_data=audio_data,
            sample_rate=sample_rate,
            duration_ms=duration_ms,
            latency_ms=latency_ms,
            model_used="VITS",
        )
    
    def synthesize_streaming(
        self,
        text: str,
        language: str = "en",
        chunk_size: int = 50,
        speed: float = 1.0,
    ) -> Generator[np.ndarray, None, None]:
        """Stream speech generation"""
        if not self._initialized:
            self.initialize()
        
        # Split text into chunks
        words = text.split()
        text_chunks = []
        current_chunk = []
        
        for word in words:
            current_chunk.append(word)
            if len(current_chunk) >= chunk_size:
                text_chunks.append(" ".join(current_chunk))
                current_chunk = []
        
        if current_chunk:
            text_chunks.append(" ".join(current_chunk))
        
        # Generate audio for each chunk
        for chunk in text_chunks:
            result = self.synthesize(chunk, language=language, speed=speed)
            yield result.audio_data


class VoiceProcessingEngine:
    """
    Main Voice Processing Engine
    Combines STT and TTS capabilities
    """
    
    def __init__(
        self,
        stt_model: str = "openai/whisper-medium",
        tts_model: str = "facebook/mms-tts-eng",
        device: str = "auto",
        precision: str = "fp16",
        load_stt_in_8bit: bool = True,
    ):
        self.stt_model_name = stt_model
        self.tts_model_name = tts_model
        self.device = self._detect_device(device)
        self.precision = precision
        self.load_stt_in_8bit = load_stt_in_8bit
        
        self.stt_backend: Optional[WhisperBackend] = None
        self.tts_backend: Optional[VITSBackend] = None
        self._stt_initialized = False
        self._tts_initialized = False
    
    def _detect_device(self, device: str) -> str:
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def initialize_stt(self):
        """Initialize speech-to-text engine"""
        if self._stt_initialized:
            return
        
        self.stt_backend = WhisperBackend(
            model_name=self.stt_model_name,
            device=self.device,
            precision=self.precision,
            load_in_8bit=self.load_stt_in_8bit,
        )
        self.stt_backend.initialize()
        self._stt_initialized = True
    
    def initialize_tts(self):
        """Initialize text-to-speech engine"""
        if self._tts_initialized:
            return
        
        self.tts_backend = VITSBackend(
            model_name=self.tts_model_name,
            device=self.device,
            precision=self.precision,
        )
        self.tts_backend.initialize()
        self._tts_initialized = True
    
    def speech_to_text(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
        **kwargs
    ) -> TranscriptionResult:
        """Convert speech to text"""
        if not self._stt_initialized:
            self.initialize_stt()
        
        return self.stt_backend.transcribe(
            audio_data=audio_data,
            sample_rate=sample_rate,
            language=language,
            **kwargs
        )
    
    def text_to_speech(
        self,
        text: str,
        language: str = "en",
        speed: float = 1.0,
        **kwargs
    ) -> SpeechGenerationResult:
        """Convert text to speech"""
        if not self._tts_initialized:
            self.initialize_tts()
        
        return self.tts_backend.synthesize(
            text=text,
            language=language,
            speed=speed,
            **kwargs
        )
    
    def translate_speech(
        self,
        audio_data: np.ndarray,
        source_language: str,
        target_language: str,
        target_speech: bool = True,
        sample_rate: int = 16000,
    ) -> Tuple[TranscriptionResult, Optional[SpeechGenerationResult], str]:
        """
        Full speech-to-speech translation pipeline
        
        Returns:
            Tuple of (transcription, generated_speech, translated_text)
        """
        # Step 1: Transcribe source speech
        transcription = self.speech_to_text(
            audio_data=audio_data,
            sample_rate=sample_rate,
            language=source_language,
        )
        
        # Step 2: Translate text (will be done by LLM engine in pipeline)
        translated_text = transcription.text  # Placeholder
        
        # Step 3: Generate target speech
        generated_speech = None
        if target_speech:
            generated_speech = self.text_to_speech(
                text=translated_text,
                language=target_language,
            )
        
        return transcription, generated_speech, translated_text
    
    def stream_speech_to_text(
        self,
        audio_chunks: Generator[np.ndarray, None, None],
        sample_rate: int = 16000,
        language: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """Stream speech-to-text from audio chunks"""
        if not self._stt_initialized:
            self.initialize_stt()
        
        yield from self.stt_backend.transcribe_streaming(
            audio_chunks=audio_chunks,
            sample_rate=sample_rate,
            language=language,
        )
    
    def stream_text_to_speech(
        self,
        text: str,
        language: str = "en",
        chunk_size: int = 50,
        speed: float = 1.0,
    ) -> Generator[np.ndarray, None, None]:
        """Stream text-to-speech generation"""
        if not self._tts_initialized:
            self.initialize_tts()
        
        yield from self.tts_backend.synthesize_streaming(
            text=text,
            language=language,
            chunk_size=chunk_size,
            speed=speed,
        )
    
    def unload(self):
        """Unload models from memory"""
        if self.stt_backend and self.stt_backend.model:
            del self.stt_backend.model
            self.stt_backend = None
            self._stt_initialized = False
        
        if self.tts_backend and self.tts_backend.model:
            del self.tts_backend.model
            self.tts_backend = None
            self._tts_initialized = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("Voice processing models unloaded")
