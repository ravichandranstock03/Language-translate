"""
Advanced Configuration Management for LLM Translator
Supports dynamic model switching, GPU optimization, and latency tuning
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Literal
from enum import Enum
import json
import os


class ModelBackend(Enum):
    """Supported LLM backends for translation"""
    NLLB = "nllb"  # Facebook NLLB-200
    MARIAN = "marian"  # MarianMT
    M2M100 = "m2m100"  # Facebook M2M100
    WHISPER = "whisper"  # OpenAI Whisper (speech-to-text)
    BARK = "bark"  # Suno Bark (text-to-speech)
    VITS = "vits"  # VITS TTS
    CUSTOM = "custom"  # Custom HuggingFace model


class PrecisionMode(Enum):
    """GPU precision modes for optimization"""
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    BF16 = "bf16"


@dataclass
class LanguageConfig:
    """Language-specific configuration"""
    code: str  # ISO 639-1 or 639-3 code
    name: str
    script: Optional[str] = None
    region: Optional[str] = None
    alternative_codes: List[str] = field(default_factory=list)
    
    # Common languages with full support
    @classmethod
    def get_supported_languages(cls) -> Dict[str, "LanguageConfig"]:
        return {
            "en": cls("en", "English", "Latin", "US"),
            "es": cls("es", "Spanish", "Latin", "ES"),
            "fr": cls("fr", "French", "Latin", "FR"),
            "de": cls("de", "German", "Latin", "DE"),
            "it": cls("it", "Italian", "Latin", "IT"),
            "pt": cls("pt", "Portuguese", "Latin", "BR"),
            "ru": cls("ru", "Russian", "Cyrillic", "RU"),
            "zh": cls("zh", "Chinese", "Han", "CN"),
            "ja": cls("ja", "Japanese", "Japanese", "JP"),
            "ko": cls("ko", "Korean", "Hangul", "KR"),
            "ar": cls("ar", "Arabic", "Arabic", "SA"),
            "hi": cls("hi", "Hindi", "Devanagari", "IN"),
            "tr": cls("tr", "Turkish", "Latin", "TR"),
            "pl": cls("pl", "Polish", "Latin", "PL"),
            "nl": cls("nl", "Dutch", "Latin", "NL"),
            "sv": cls("sv", "Swedish", "Latin", "SE"),
            "da": cls("da", "Danish", "Latin", "DK"),
            "fi": cls("fi", "Finnish", "Latin", "FI"),
            "no": cls("no", "Norwegian", "Latin", "NO"),
            "cs": cls("cs", "Czech", "Latin", "CZ"),
            "el": cls("el", "Greek", "Greek", "GR"),
            "he": cls("he", "Hebrew", "Hebrew", "IL"),
            "th": cls("th", "Thai", "Thai", "TH"),
            "vi": cls("vi", "Vietnamese", "Latin", "VN"),
            "id": cls("id", "Indonesian", "Latin", "ID"),
            "ms": cls("ms", "Malay", "Latin", "MY"),
            "tl": cls("tl", "Tagalog", "Latin", "PH"),
            "uk": cls("uk", "Ukrainian", "Cyrillic", "UA"),
            "bg": cls("bg", "Bulgarian", "Cyrillic", "BG"),
            "hr": cls("hr", "Croatian", "Latin", "HR"),
            "sk": cls("sk", "Slovak", "Latin", "SK"),
            "sl": cls("sl", "Slovenian", "Latin", "SI"),
            "sr": cls("sr", "Serbian", "Cyrillic", "RS"),
            "ro": cls("ro", "Romanian", "Latin", "RO"),
            "hu": cls("hu", "Hungarian", "Latin", "HU"),
            "ca": cls("ca", "Catalan", "Latin", "ES"),
            "fa": cls("fa", "Persian", "Arabic", "IR"),
            "ur": cls("ur", "Urdu", "Arabic", "PK"),
            "bn": cls("bn", "Bengali", "Bengali", "BD"),
            "ta": cls("ta", "Tamil", "Tamil", "IN"),
            "te": cls("te", "Telugu", "Telugu", "IN"),
            "mr": cls("mr", "Marathi", "Devanagari", "IN"),
            "gu": cls("gu", "Gujarati", "Gujarati", "IN"),
            "kn": cls("kn", "Kannada", "Kannada", "IN"),
            "ml": cls("ml", "Malayalam", "Malayalam", "IN"),
            "pa": cls("pa", "Punjabi", "Gurmukhi", "IN"),
        }


@dataclass
class PerformanceConfig:
    """Performance and latency optimization settings"""
    max_batch_size: int = 32
    beam_size: int = 4
    max_length: int = 512
    min_length: int = 0
    length_penalty: float = 1.0
    repetition_penalty: float = 1.1
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    do_sample: bool = False
    
    # Latency optimization
    enable_streaming: bool = True
    chunk_size: int = 1024  # Audio chunk size in samples
    overlap_size: int = 256  # Overlap for smooth transitions
    max_latency_ms: int = 200  # Target maximum latency


@dataclass
class HardwareConfig:
    """Hardware acceleration configuration"""
    device: str = "auto"  # auto, cuda, mps, cpu
    precision: PrecisionMode = PrecisionMode.FP16
    gpu_memory_fraction: float = 0.9
    enable_tf32: bool = True
    enable_cudnn_benchmark: bool = True
    num_threads: int = -1  # -1 for auto
    num_workers: int = 4
    
    def get_device(self) -> str:
        """Auto-detect best available device"""
        if self.device != "auto":
            return self.device
        
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"


@dataclass
class ModelConfig:
    """Individual model configuration"""
    backend: ModelBackend
    model_name: str
    revision: Optional[str] = None
    cache_dir: Optional[str] = None
    trust_remote_code: bool = False
    use_auth_token: Optional[str] = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    device_map: Optional[str] = "auto"


@dataclass
class Config:
    """Main configuration class for the translator system"""
    
    # System identification
    name: str = "RealTimeLLMTranslator"
    version: str = "1.0.0"
    
    # Language settings
    source_language: str = "auto"  # 'auto' for auto-detection
    target_language: str = "en"
    supported_languages: Dict[str, LanguageConfig] = field(default_factory=dict)
    
    # Model configurations
    text_translation_model: Optional[ModelConfig] = None
    speech_to_text_model: Optional[ModelConfig] = None
    text_to_speech_model: Optional[ModelConfig] = None
    
    # Performance settings
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    # Hardware settings
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_debug: bool = False
    max_concurrent_requests: int = 10
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    def __post_init__(self):
        """Initialize default models if not specified"""
        if not self.supported_languages:
            self.supported_languages = LanguageConfig.get_supported_languages()
        
        if self.text_translation_model is None:
            self.text_translation_model = ModelConfig(
                backend=ModelBackend.NLLB,
                model_name="facebook/nllb-200-distilled-600M",
                load_in_8bit=True,
            )
        
        if self.speech_to_text_model is None:
            self.speech_to_text_model = ModelConfig(
                backend=ModelBackend.WHISPER,
                model_name="openai/whisper-medium",
                load_in_8bit=True,
            )
        
        if self.text_to_speech_model is None:
            self.text_to_speech_model = ModelConfig(
                backend=ModelBackend.VITS,
                model_name="facebook/mms-tts-eng",
            )
    
    def save(self, filepath: str):
        """Save configuration to JSON file"""
        config_dict = {
            "name": self.name,
            "version": self.version,
            "source_language": self.source_language,
            "target_language": self.target_language,
            "performance": {
                "max_batch_size": self.performance.max_batch_size,
                "beam_size": self.performance.beam_size,
                "enable_streaming": self.performance.enable_streaming,
                "max_latency_ms": self.performance.max_latency_ms,
            },
            "hardware": {
                "device": self.hardware.device,
                "precision": self.hardware.precision.value,
                "num_workers": self.hardware.num_workers,
            },
            "api": {
                "host": self.api_host,
                "port": self.api_port,
                "max_concurrent_requests": self.max_concurrent_requests,
            },
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> "Config":
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        return cls(
            name=config_dict.get("name", "RealTimeLLMTranslator"),
            version=config_dict.get("version", "1.0.0"),
            source_language=config_dict.get("source_language", "auto"),
            target_language=config_dict.get("target_language", "en"),
            # Additional fields can be populated as needed
        )
    
    def optimize_for_low_latency(self):
        """Apply aggressive optimizations for minimal latency"""
        self.performance.max_batch_size = 1
        self.performance.beam_size = 1
        self.performance.enable_streaming = True
        self.performance.chunk_size = 512
        self.performance.overlap_size = 128
        self.performance.max_latency_ms = 100
        self.performance.do_sample = False
        self.hardware.num_workers = 8
    
    def optimize_for_quality(self):
        """Apply optimizations for highest translation quality"""
        self.performance.max_batch_size = 8
        self.performance.beam_size = 5
        self.performance.length_penalty = 1.2
        self.performance.repetition_penalty = 1.2
        self.performance.max_latency_ms = 500
