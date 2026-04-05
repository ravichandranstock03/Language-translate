"""
Real-Time LLM-Based Voice & Text Translator
Advanced Architecture with Streaming Support
"""

__version__ = "1.0.0"
__author__ = "Advanced LLM Systems"

from .core.config import Config, LanguageConfig
from .core.logger import setup_logger
from .engines.llm_engine import LLMTranslationEngine
from .engines.voice_engine import VoiceProcessingEngine
from .engines.streaming_engine import StreamingEngine
from .pipeline.translation_pipeline import TranslationPipeline
from .api.server import TranslationAPI

__all__ = [
    "Config",
    "LanguageConfig",
    "setup_logger",
    "LLMTranslationEngine",
    "VoiceProcessingEngine",
    "StreamingEngine",
    "TranslationPipeline",
    "TranslationAPI",
]
