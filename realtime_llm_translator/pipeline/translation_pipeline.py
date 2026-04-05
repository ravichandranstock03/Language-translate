"""
Advanced Translation Pipeline
Coordinates LLM, Voice, and Streaming engines for end-to-end translation
"""

import torch
import numpy as np
from typing import Optional, Dict, List, Generator, Tuple, Union, Callable
from dataclasses import dataclass, field
import threading
import time
from enum import Enum
import asyncio

from ..core.config import Config
from ..core.logger import setup_logger
from ..engines.llm_engine import LLMTranslationEngine, TranslationResult
from ..engines.voice_engine import VoiceProcessingEngine, TranscriptionResult, SpeechGenerationResult
from ..engines.streaming_engine import StreamingEngine, StreamConfig, AudioChunk


class PipelineMode(Enum):
    """Translation pipeline modes"""
    TEXT_ONLY = "text_only"
    VOICE_TO_TEXT = "voice_to_text"
    TEXT_TO_VOICE = "text_to_voice"
    VOICE_TO_VOICE = "voice_to_voice"
    REALTIME_STREAMING = "realtime_streaming"


@dataclass
class PipelineResult:
    """Container for complete pipeline results"""
    # Input
    input_text: Optional[str] = None
    input_audio: Optional[np.ndarray] = None
    source_language: str = "auto"
    
    # Output
    translated_text: Optional[str] = None
    output_audio: Optional[np.ndarray] = None
    target_language: str = "en"
    
    # Intermediate results
    transcription: Optional[TranscriptionResult] = None
    translation_result: Optional[TranslationResult] = None
    speech_generation: Optional[SpeechGenerationResult] = None
    
    # Metrics
    total_latency_ms: float = 0.0
    stages_completed: List[str] = field(default_factory=list)
    mode: PipelineMode = PipelineMode.TEXT_ONLY
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    success: bool = True
    error_message: Optional[str] = None


class TranslationPipeline:
    """
    Advanced Translation Pipeline
    Orchestrates all components for seamless translation experience
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        mode: PipelineMode = PipelineMode.TEXT_ONLY,
        on_translation_update: Optional[Callable[[str], None]] = None,
        on_audio_ready: Optional[Callable[[np.ndarray], None]] = None,
    ):
        self.config = config or Config()
        self.mode = mode
        self.on_translation_update = on_translation_update
        self.on_audio_ready = on_audio_ready
        
        # Initialize logger
        self.logger = setup_logger(
            name="TranslationPipeline",
            level=self.config.log_level,
            log_file=self.config.log_file,
        )
        
        # Initialize engines (lazy loading)
        self.llm_engine: Optional[LLMTranslationEngine] = None
        self.voice_engine: Optional[VoiceProcessingEngine] = None
        self.streaming_engine: Optional[StreamingEngine] = None
        
        self._initialized = False
        self._lock = threading.Lock()
        
        # Performance tracking
        self.total_translations = 0
        self.total_errors = 0
        self.latency_history: List[float] = []
    
    def initialize(self):
        """Initialize all required engines based on mode"""
        with self._lock:
            if self._initialized:
                return
            
            self.logger.info(f"Initializing pipeline in {self.mode.value} mode")
            
            # Initialize LLM engine for text translation
            if self.mode in [
                PipelineMode.TEXT_ONLY,
                PipelineMode.VOICE_TO_TEXT,
                PipelineMode.TEXT_TO_VOICE,
                PipelineMode.VOICE_TO_VOICE,
                PipelineMode.REALTIME_STREAMING,
            ]:
                model_config = self.config.text_translation_model
                self.llm_engine = LLMTranslationEngine(
                    backend=model_config.backend.value,
                    model_name=model_config.model_name,
                    device=self.config.hardware.get_device(),
                    precision=self.config.hardware.precision.value,
                    cache_dir=model_config.cache_dir,
                    load_in_8bit=model_config.load_in_8bit,
                )
                self.llm_engine.initialize()
                self.logger.info("LLM translation engine initialized")
            
            # Initialize voice engine for STT/TTS
            if self.mode in [
                PipelineMode.VOICE_TO_TEXT,
                PipelineMode.TEXT_TO_VOICE,
                PipelineMode.VOICE_TO_VOICE,
                PipelineMode.REALTIME_STREAMING,
            ]:
                stt_config = self.config.speech_to_text_model
                tts_config = self.config.text_to_speech_model
                
                self.voice_engine = VoiceProcessingEngine(
                    stt_model=stt_config.model_name,
                    tts_model=tts_config.model_name,
                    device=self.config.hardware.get_device(),
                    precision=self.config.hardware.precision.value,
                    load_stt_in_8bit=stt_config.load_in_8bit,
                )
                
                if self.mode in [PipelineMode.VOICE_TO_TEXT, PipelineMode.VOICE_TO_VOICE, PipelineMode.REALTIME_STREAMING]:
                    self.voice_engine.initialize_stt()
                    self.logger.info("Speech-to-text engine initialized")
                
                if self.mode in [PipelineMode.TEXT_TO_VOICE, PipelineMode.VOICE_TO_VOICE, PipelineMode.REALTIME_STREAMING]:
                    self.voice_engine.initialize_tts()
                    self.logger.info("Text-to-speech engine initialized")
            
            # Initialize streaming engine for real-time mode
            if self.mode == PipelineMode.REALTIME_STREAMING:
                stream_config = StreamConfig(
                    sample_rate=16000,
                    chunk_size_ms=self.config.performance.chunk_size,
                    overlap_ms=self.config.performance.overlap_size,
                )
                
                self.streaming_engine = StreamingEngine(
                    config=stream_config,
                    on_translation_update=self.on_translation_update,
                    on_audio_ready=self.on_audio_ready,
                )
                self.logger.info("Streaming engine initialized")
            
            self._initialized = True
            self.logger.info("Pipeline initialization complete")
    
    def translate_text(
        self,
        text: str,
        source_language: str,
        target_language: str,
        streaming: bool = False,
    ) -> Union[PipelineResult, Generator[str, None, None]]:
        """
        Translate text from source to target language
        
        Args:
            text: Input text to translate
            source_language: Source language code
            target_language: Target language code
            streaming: If True, yield partial translations
        
        Returns:
            PipelineResult or generator of partial translations
        """
        start_time = time.time()
        
        if not self._initialized:
            self.initialize()
        
        try:
            self.logger.info(f"Translating text from {source_language} to {target_language}")
            
            if streaming:
                def generate_stream():
                    partial_text = ""
                    for chunk in self.llm_engine.translate_streaming(
                        text=text,
                        source_language=source_language,
                        target_language=target_language,
                        chunk_size=20,
                    ):
                        partial_text = chunk
                        yield chunk
                        
                        if self.on_translation_update:
                            self.on_translation_update(chunk)
                
                return generate_stream()
            else:
                result = self.llm_engine.translate(
                    text=text,
                    source_language=source_language,
                    target_language=target_language,
                    max_length=self.config.performance.max_length,
                    num_beams=self.config.performance.beam_size,
                    temperature=self.config.performance.temperature,
                )
                
                total_latency = (time.time() - start_time) * 1000
                
                pipeline_result = PipelineResult(
                    input_text=text,
                    translated_text=result.translated_text,
                    source_language=source_language,
                    target_language=target_language,
                    translation_result=result,
                    total_latency_ms=total_latency,
                    stages_completed=["translation"],
                    mode=PipelineMode.TEXT_ONLY,
                    success=True,
                )
                
                # Update metrics
                self.total_translations += 1
                self.latency_history.append(total_latency)
                
                # Log metrics
                self.logger.metrics.log_latency(
                    "text_translation",
                    total_latency,
                    {"source": source_language, "target": target_language}
                )
                
                if self.on_translation_update:
                    self.on_translation_update(result.translated_text)
                
                return pipeline_result
        
        except Exception as e:
            self.total_errors += 1
            self.logger.error(f"Translation error: {e}")
            
            return PipelineResult(
                input_text=text,
                source_language=source_language,
                target_language=target_language,
                total_latency_ms=(time.time() - start_time) * 1000,
                stages_completed=[],
                mode=PipelineMode.TEXT_ONLY,
                success=False,
                error_message=str(e),
            )
    
    def translate_speech(
        self,
        audio_data: np.ndarray,
        source_language: str,
        target_language: str,
        output_speech: bool = True,
        sample_rate: int = 16000,
    ) -> PipelineResult:
        """
        Translate speech to text or speech
        
        Args:
            audio_data: Input audio waveform
            source_language: Source language code
            target_language: Target language code
            output_speech: If True, generate translated speech
            sample_rate: Audio sample rate
        
        Returns:
            PipelineResult with transcription, translation, and optionally synthesized speech
        """
        start_time = time.time()
        
        if not self._initialized:
            self.initialize()
        
        try:
            self.logger.info(f"Translating speech from {source_language} to {target_language}")
            
            stages_completed = []
            
            # Stage 1: Speech-to-Text
            transcription = self.voice_engine.speech_to_text(
                audio_data=audio_data,
                sample_rate=sample_rate,
                language=source_language,
            )
            stages_completed.append("speech_to_text")
            
            self.logger.info(f"Transcribed: {transcription.text}")
            
            # Stage 2: Text Translation
            translation_result = self.llm_engine.translate(
                text=transcription.text,
                source_language=source_language,
                target_language=target_language,
                max_length=self.config.performance.max_length,
                num_beams=self.config.performance.beam_size,
            )
            stages_completed.append("translation")
            
            self.logger.info(f"Translated: {translation_result.translated_text}")
            
            # Stage 3: Text-to-Speech (optional)
            speech_generation = None
            if output_speech:
                speech_generation = self.voice_engine.text_to_speech(
                    text=translation_result.translated_text,
                    language=target_language,
                )
                stages_completed.append("text_to_speech")
                
                if self.on_audio_ready:
                    self.on_audio_ready(speech_generation.audio_data)
            
            total_latency = (time.time() - start_time) * 1000
            
            pipeline_result = PipelineResult(
                input_audio=audio_data,
                translated_text=translation_result.translated_text,
                output_audio=speech_generation.audio_data if speech_generation else None,
                source_language=source_language,
                target_language=target_language,
                transcription=transcription,
                translation_result=translation_result,
                speech_generation=speech_generation,
                total_latency_ms=total_latency,
                stages_completed=stages_completed,
                mode=PipelineMode.VOICE_TO_VOICE if output_speech else PipelineMode.VOICE_TO_TEXT,
                success=True,
            )
            
            # Update metrics
            self.total_translations += 1
            self.latency_history.append(total_latency)
            
            self.logger.metrics.log_latency(
                "speech_translation",
                total_latency,
                {
                    "source": source_language,
                    "target": target_language,
                    "output_speech": output_speech,
                }
            )
            
            return pipeline_result
        
        except Exception as e:
            self.total_errors += 1
            self.logger.error(f"Speech translation error: {e}")
            
            return PipelineResult(
                input_audio=audio_data,
                source_language=source_language,
                target_language=target_language,
                total_latency_ms=(time.time() - start_time) * 1000,
                stages_completed=[],
                mode=PipelineMode.VOICE_TO_VOICE,
                success=False,
                error_message=str(e),
            )
    
    def translate_streaming(
        self,
        audio_chunks: Generator[np.ndarray, None, None],
        source_language: str,
        target_language: str,
        output_speech: bool = True,
    ) -> Generator[PipelineResult, None, None]:
        """
        Real-time streaming translation
        
        Args:
            audio_chunks: Generator of audio chunks
            source_language: Source language code
            target_language: Target language code
            output_speech: If True, generate translated speech
        
        Yields:
            PipelineResult for each processed chunk
        """
        if not self._initialized:
            self.initialize()
        
        self.logger.info("Starting real-time streaming translation")
        
        # Start streaming engine
        self.streaming_engine.start()
        
        try:
            # Push audio chunks to streaming engine
            for chunk in audio_chunks:
                self.streaming_engine.push_audio(chunk)
            
            # Process chunks as they become available
            for audio_chunk in self.streaming_engine.get_audio_chunks():
                if audio_chunk.is_speech:
                    # Transcribe chunk
                    transcription = self.voice_engine.speech_to_text(
                        audio_data=audio_chunk.data,
                        sample_rate=self.streaming_engine.config.sample_rate,
                        language=source_language,
                    )
                    
                    # Translate text
                    translation = self.llm_engine.translate(
                        text=transcription.text,
                        source_language=source_language,
                        target_language=target_language,
                    )
                    
                    # Generate speech if needed
                    speech = None
                    if output_speech:
                        speech = self.voice_engine.text_to_speech(
                            text=translation.translated_text,
                            language=target_language,
                        )
                        
                        if self.on_audio_ready:
                            self.on_audio_ready(speech.audio_data)
                    
                    result = PipelineResult(
                        input_audio=audio_chunk.data,
                        translated_text=translation.translated_text,
                        output_audio=speech.audio_data if speech else None,
                        source_language=source_language,
                        target_language=target_language,
                        transcription=transcription,
                        translation_result=translation,
                        speech_generation=speech,
                        total_latency_ms=translation.latency_ms,
                        stages_completed=["stt", "translation"] + (["tts"] if speech else []),
                        mode=PipelineMode.REALTIME_STREAMING,
                        success=True,
                    )
                    
                    yield result
                    
                    if self.on_translation_update:
                        self.on_translation_update(translation.translated_text)
        
        finally:
            self.streaming_engine.stop()
            self.logger.info("Streaming translation completed")
    
    async def translate_async(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> PipelineResult:
        """Async wrapper for text translation"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.translate_text(text, source_language, target_language)
        )
    
    def get_stats(self) -> Dict:
        """Get pipeline performance statistics"""
        avg_latency = sum(self.latency_history) / len(self.latency_history) if self.latency_history else 0
        max_latency = max(self.latency_history) if self.latency_history else 0
        min_latency = min(self.latency_history) if self.latency_history else 0
        
        return {
            "total_translations": self.total_translations,
            "total_errors": self.total_errors,
            "success_rate": self.total_translations / (self.total_translations + self.total_errors) if (self.total_translations + self.total_errors) > 0 else 0,
            "avg_latency_ms": avg_latency,
            "max_latency_ms": max_latency,
            "min_latency_ms": min_latency,
            "mode": self.mode.value,
            "initialized": self._initialized,
        }
    
    def unload(self):
        """Unload all models and release resources"""
        self.logger.info("Unloading pipeline models")
        
        if self.llm_engine:
            self.llm_engine.unload()
        
        if self.voice_engine:
            self.voice_engine.unload()
        
        if self.streaming_engine:
            self.streaming_engine.stop()
        
        self._initialized = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("Pipeline unloaded successfully")
