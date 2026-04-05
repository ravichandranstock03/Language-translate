"""
FastAPI-based REST and WebSocket API for Real-time Translation
Provides endpoints for text translation, speech-to-text, text-to-speech, and streaming
"""

import asyncio
import base64
import numpy as np
from typing import Optional, Dict, List, AsyncGenerator
from fastapi import FastAPI, WebSocket, HTTPException, BackgroundTasks, File, UploadFile, Form
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import io
import wave
import json
import time

from ..core.config import Config
from ..core.logger import setup_logger
from ..pipeline.translation_pipeline import TranslationPipeline, PipelineMode, PipelineResult


# Pydantic models for request/response
class TextTranslationRequest(BaseModel):
    """Request model for text translation"""
    text: str = Field(..., min_length=1, max_length=5000)
    source_language: str = Field(default="auto", description="Source language code")
    target_language: str = Field(default="en", description="Target language code")
    streaming: bool = Field(default=False, description="Enable streaming response")


class TextTranslationResponse(BaseModel):
    """Response model for text translation"""
    translated_text: str
    source_language: str
    target_language: str
    latency_ms: float
    confidence: float
    success: bool
    error_message: Optional[str] = None


class SpeechTranslationRequest(BaseModel):
    """Request model for speech translation"""
    source_language: str = Field(default="auto")
    target_language: str = Field(default="en")
    output_speech: bool = Field(default=True)


class StreamMessage(BaseModel):
    """WebSocket stream message"""
    type: str  # "audio", "text", "control"
    data: Optional[str] = None
    metadata: Optional[Dict] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    mode: str
    initialized: bool
    total_translations: int
    avg_latency_ms: float


class TranslationAPI:
    """
    FastAPI-based Translation API
    Provides REST and WebSocket endpoints for all translation modes
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        mode: PipelineMode = PipelineMode.TEXT_ONLY,
    ):
        self.config = config or Config()
        self.mode = mode
        self.logger = setup_logger(
            name="TranslationAPI",
            level=self.config.log_level,
        )
        
        # Initialize pipeline
        self.pipeline = TranslationPipeline(
            config=self.config,
            mode=self.mode,
        )
        
        # Create FastAPI app
        self.app = self._create_app()
        
        # Active WebSocket connections
        self.active_connections: List[WebSocket] = []
    
    def _create_app(self) -> FastAPI:
        """Create and configure FastAPI application"""
        app = FastAPI(
            title="Real-Time LLM Translator API",
            description="Advanced real-time translation API with LLM support",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc",
        )
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Register routes
        @app.get("/", tags=["Root"])
        async def root():
            return {
                "name": "Real-Time LLM Translator API",
                "version": "1.0.0",
                "status": "running"
            }
        
        @app.get("/health", response_model=HealthResponse, tags=["Health"])
        async def health_check():
            stats = self.pipeline.get_stats()
            return HealthResponse(
                status="healthy",
                version="1.0.0",
                mode=stats["mode"],
                initialized=stats["initialized"],
                total_translations=stats["total_translations"],
                avg_latency_ms=stats["avg_latency_ms"],
            )
        
        @app.post(
            "/translate/text",
            response_model=TextTranslationResponse,
            tags=["Text Translation"]
        )
        async def translate_text(request: TextTranslationRequest):
            """Translate text from source to target language"""
            try:
                result = self.pipeline.translate_text(
                    text=request.text,
                    source_language=request.source_language,
                    target_language=request.target_language,
                    streaming=request.streaming,
                )
                
                if isinstance(result, PipelineResult):
                    return TextTranslationResponse(
                        translated_text=result.translated_text,
                        source_language=result.source_language,
                        target_language=result.target_language,
                        latency_ms=result.total_latency_ms,
                        confidence=result.translation_result.confidence_score if result.translation_result else 0.0,
                        success=result.success,
                        error_message=result.error_message,
                    )
                else:
                    # Streaming response
                    async def generate():
                        for chunk in result:
                            yield f"data: {json.dumps({'text': chunk})}\n\n"
                    
                    return StreamingResponse(
                        generate(),
                        media_type="text/event-stream",
                    )
            
            except Exception as e:
                self.logger.error(f"Text translation error: {e}")
                return TextTranslationResponse(
                    translated_text="",
                    source_language=request.source_language,
                    target_language=request.target_language,
                    latency_ms=0,
                    confidence=0.0,
                    success=False,
                    error_message=str(e),
                )
        
        @app.post("/translate/speech", tags=["Speech Translation"])
        async def translate_speech(
            file: UploadFile = File(...),
            source_language: str = Form(default="auto"),
            target_language: str = Form(default="en"),
            output_speech: bool = Form(default=True),
        ):
            """Translate speech audio file"""
            try:
                # Read audio file
                contents = await file.read()
                
                # Parse WAV file
                with io.BytesIO(contents) as wav_file:
                    with wave.open(wav_file, 'rb') as wav:
                        sample_rate = wav.getframerate()
                        n_channels = wav.getnchannels()
                        sample_width = wav.getsampwidth()
                        n_frames = wav.getnframes()
                        
                        audio_data = np.frombuffer(
                            wav.readframes(n_frames),
                            dtype=np.int16 if sample_width == 2 else np.float32
                        )
                        
                        if n_channels > 1:
                            audio_data = audio_data.reshape(-1, n_channels).mean(axis=1)
                
                # Translate
                result = self.pipeline.translate_speech(
                    audio_data=audio_data.astype(np.float32),
                    source_language=source_language,
                    target_language=target_language,
                    output_speech=output_speech,
                    sample_rate=sample_rate,
                )
                
                if result.success:
                    response_data = {
                        "transcription": result.transcription.text if result.transcription else "",
                        "translation": result.translated_text,
                        "latency_ms": result.total_latency_ms,
                        "stages_completed": result.stages_completed,
                    }
                    
                    # Include audio if requested
                    if result.output_audio is not None:
                        # Convert to WAV format
                        output_buffer = io.BytesIO()
                        with wave.open(output_buffer, 'wb') as wav_out:
                            wav_out.setnchannels(1)
                            wav_out.setsampwidth(2)
                            wav_out.setframerate(16000)
                            wav_out.writeframes((result.output_audio * 32767).astype(np.int16).tobytes())
                        
                        audio_base64 = base64.b64encode(output_buffer.getvalue()).decode()
                        response_data["output_audio"] = audio_base64
                    
                    return JSONResponse(content=response_data)
                else:
                    raise HTTPException(status_code=500, detail=result.error_message)
            
            except Exception as e:
                self.logger.error(f"Speech translation error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.websocket("/ws/stream")
        async def websocket_stream(websocket: WebSocket):
            """WebSocket endpoint for real-time streaming translation"""
            await websocket.accept()
            self.active_connections.append(websocket)
            
            try:
                self.logger.info("New WebSocket connection established")
                
                # Initialize streaming
                await websocket.send_json({
                    "type": "status",
                    "data": "connected",
                    "metadata": {"mode": self.mode.value}
                })
                
                # Handle streaming
                audio_chunks = []
                
                async def audio_generator():
                    while True:
                        try:
                            message = await websocket.receive_json()
                            
                            if message.get("type") == "audio":
                                # Decode base64 audio
                                audio_data = base64.b64decode(message["data"])
                                audio_array = np.frombuffer(audio_data, dtype=np.float32)
                                audio_chunks.append(audio_array)
                                
                            elif message.get("type") == "control":
                                if message.get("data") == "stop":
                                    break
                        
                        except Exception as e:
                            self.logger.error(f"WebSocket receive error: {e}")
                            break
                
                # Process audio chunks
                async def process_and_send():
                    for chunk in audio_chunks:
                        try:
                            # Transcribe
                            transcription = self.pipeline.voice_engine.speech_to_text(
                                audio_data=chunk,
                                sample_rate=16000,
                                language="auto",
                            )
                            
                            # Translate
                            translation = self.pipeline.llm_engine.translate(
                                text=transcription.text,
                                source_language="auto",
                                target_language=self.config.target_language,
                            )
                            
                            # Send result
                            await websocket.send_json({
                                "type": "translation",
                                "data": translation.translated_text,
                                "metadata": {
                                    "latency_ms": translation.latency_ms,
                                    "original": transcription.text,
                                }
                            })
                            
                        except Exception as e:
                            self.logger.error(f"Processing error: {e}")
                            await websocket.send_json({
                                "type": "error",
                                "data": str(e),
                            })
                
                # Run processing
                await asyncio.gather(
                    audio_generator(),
                    process_and_send(),
                )
            
            finally:
                self.active_connections.remove(websocket)
                await websocket.close()
                self.logger.info("WebSocket connection closed")
        
        @app.get("/stats", tags=["Monitoring"])
        async def get_stats():
            """Get pipeline performance statistics"""
            return self.pipeline.get_stats()
        
        @app.post("/unload", tags=["Administration"])
        async def unload_models():
            """Unload models from memory"""
            self.pipeline.unload()
            return {"status": "models_unloaded"}
        
        return app
    
    def run(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        reload: bool = False,
        log_level: str = "info",
    ):
        """Run the API server"""
        self.logger.info(f"Starting API server on {host}:{port}")
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            reload=reload,
            log_level=log_level,
        )


def create_app(
    config: Optional[Config] = None,
    mode: PipelineMode = PipelineMode.TEXT_ONLY,
) -> FastAPI:
    """Factory function to create FastAPI app"""
    api = TranslationAPI(config=config, mode=mode)
    return api.app
