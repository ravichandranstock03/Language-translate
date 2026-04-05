"""
Advanced LLM Translation Engine
Supports multiple backends: NLLB, MarianMT, M2M100 with streaming capabilities
"""

import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    MarianMTModel,
    MarianTokenizer,
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
    NllbForConditionalGeneration,
    NllbTokenizer,
    pipeline,
    Pipeline,
)
from typing import Optional, Dict, List, Generator, Tuple, Union
from abc import ABC, abstractmethod
import time
from dataclasses import dataclass
import threading
from queue import Queue


@dataclass
class TranslationResult:
    """Container for translation results with metadata"""
    translated_text: str
    source_language: str
    target_language: str
    confidence_score: float
    latency_ms: float
    model_used: str
    tokens_processed: int
    streaming: bool = False


class BaseModelBackend(ABC):
    """Abstract base class for translation model backends"""
    
    def __init__(self, model_name: str, device: str, precision: str):
        self.model_name = model_name
        self.device = device
        self.precision = precision
        self.model = None
        self.tokenizer = None
        self._lock = threading.Lock()
    
    @abstractmethod
    def load_model(self, **kwargs):
        """Load the model and tokenizer"""
        pass
    
    @abstractmethod
    def translate(self, text: str, src_lang: str, tgt_lang: str, **kwargs) -> TranslationResult:
        """Perform translation"""
        pass
    
    @abstractmethod
    def translate_streaming(
        self, 
        text: str, 
        src_lang: str, 
        tgt_lang: str, 
        chunk_size: int = 50,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream translation output token by token"""
        pass
    
    def get_language_code(self, lang: str, target_backend: str) -> str:
        """Convert ISO language codes to backend-specific codes"""
        # Common mappings - can be extended
        code_mappings = {
            "nllb": {
                "zh": "zho_Hans",
                "zh-tw": "zho_Hant",
                "en": "eng_Latn",
                "es": "spa_Latn",
                "fr": "fra_Latn",
                "de": "deu_Latn",
                "it": "ita_Latn",
                "pt": "por_Latn",
                "ru": "rus_Cyrl",
                "ja": "jpn_Jpan",
                "ko": "kor_Hang",
                "ar": "arb_Arab",
                "hi": "hin_Deva",
                "tr": "tur_Latn",
                "pl": "pol_Latn",
                "nl": "nld_Latn",
                "sv": "swe_Latn",
                "da": "dan_Latn",
                "fi": "fin_Latn",
                "no": "nob_Latn",
                "cs": "ces_Latn",
                "el": "ell_Grek",
                "he": "heb_Hebr",
                "th": "tha_Thai",
                "vi": "vie_Latn",
                "id": "ind_Latn",
                "ms": "zsm_Latn",
                "tl": "tgl_Latn",
                "uk": "ukr_Cyrl",
                "bg": "bul_Cyrl",
                "hr": "hrv_Latn",
                "sk": "slk_Latn",
                "sl": "slv_Latn",
                "sr": "srp_Cyrl",
                "ro": "ron_Latn",
                "hu": "hun_Latn",
                "ca": "cat_Latn",
                "fa": "fas_Arab",
                "ur": "urd_Arab",
                "bn": "ben_Beng",
                "ta": "tam_Taml",
                "te": "tel_Telu",
                "mr": "mar_Deva",
                "gu": "guj_Gujr",
                "kn": "kan_Knda",
                "ml": "mal_Mlym",
                "pa": "pan_Guru",
            },
            "m2m100": {
                "en": "en",
                "es": "es",
                "fr": "fr",
                "de": "de",
                "it": "it",
                "pt": "pt",
                "ru": "ru",
                "zh": "zh",
                "ja": "ja",
                "ko": "ko",
                "ar": "ar",
                "hi": "hi",
                "tr": "tr",
                "pl": "pl",
                "nl": "nl",
                "sv": "sv",
                "da": "da",
                "fi": "fi",
                "no": "no",
                "cs": "cs",
                "el": "el",
                "he": "he",
                "th": "th",
                "vi": "vi",
                "id": "id",
                "ms": "ms",
                "tl": "tl",
                "uk": "uk",
                "bg": "bg",
                "hr": "hr",
                "sk": "sk",
                "sl": "sl",
                "sr": "sr",
                "ro": "ro",
                "hu": "hu",
                "ca": "ca",
                "fa": "fa",
                "ur": "ur",
                "bn": "bn",
                "ta": "ta",
                "te": "te",
                "mr": "mr",
                "gu": "gu",
                "kn": "kn",
                "ml": "ml",
                "pa": "pa",
            }
        }
        
        if target_backend in code_mappings:
            return code_mappings[target_backend].get(lang, lang)
        return lang


class NLLBBackend(BaseModelBackend):
    """Facebook NLLB-200 backend for high-quality translation"""
    
    def load_model(self, load_in_8bit: bool = False, cache_dir: Optional[str] = None):
        """Load NLLB model"""
        print(f"Loading NLLB model: {self.model_name} on {self.device}")
        
        dtype = torch.float16 if self.precision == "fp16" else torch.float32
        
        self.tokenizer = NllbTokenizer.from_pretrained(
            self.model_name,
            cache_dir=cache_dir,
        )
        
        self.model = NllbForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            load_in_8bit=load_in_8bit,
            cache_dir=cache_dir,
            device_map="auto" if self.device == "cuda" else None,
        )
        
        if not load_in_8bit and self.device == "cuda":
            self.model.to(self.device)
        
        self.model.eval()
        print("NLLB model loaded successfully")
    
    def translate(
        self, 
        text: str, 
        src_lang: str, 
        tgt_lang: str,
        max_length: int = 512,
        num_beams: int = 4,
        temperature: float = 1.0,
        **kwargs
    ) -> TranslationResult:
        """Translate text using NLLB"""
        start_time = time.time()
        
        src_code = self.get_language_code(src_lang, "nllb")
        tgt_code = self.get_language_code(tgt_lang, "nllb")
        
        self.tokenizer.src_lang = src_code
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt_code],
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                **kwargs
            )
        
        translated_text = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )[0]
        
        latency_ms = (time.time() - start_time) * 1000
        
        return TranslationResult(
            translated_text=translated_text,
            source_language=src_lang,
            target_language=tgt_lang,
            confidence_score=0.95,  # Placeholder - can be computed from logits
            latency_ms=latency_ms,
            model_used="NLLB",
            tokens_processed=len(generated_tokens[0]),
        )
    
    def translate_streaming(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
        chunk_size: int = 50,
        max_length: int = 512,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream translation output"""
        src_code = self.get_language_code(src_lang, "nllb")
        tgt_code = self.get_language_code(tgt_lang, "nllb")
        
        self.tokenizer.src_lang = src_code
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Use incremental decoding for streaming
            generated_ids = []
            past_key_values = None
            
            for i in range(max_length):
                if past_key_values is not None:
                    outputs = self.model(
                        input_ids=torch.tensor([[generated_ids[-1]]], device=self.device),
                        past_key_values=past_key_values,
                    )
                else:
                    outputs = self.model(**inputs)
                
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1)
                
                generated_ids.append(next_token.item())
                past_key_values = outputs.past_key_values
                
                # Check for EOS token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                # Yield partial translation every chunk_size tokens
                if len(generated_ids) % chunk_size == 0:
                    partial_text = self.tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                    yield partial_text
        
        final_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        yield final_text


class MarianMTBackend(BaseModelBackend):
    """MarianMT backend for fast translation"""
    
    def load_model(self, cache_dir: Optional[str] = None):
        """Load MarianMT model"""
        print(f"Loading MarianMT model: {self.model_name} on {self.device}")
        
        dtype = torch.float16 if self.precision == "fp16" else torch.float32
        
        self.tokenizer = MarianTokenizer.from_pretrained(
            self.model_name,
            cache_dir=cache_dir,
        )
        
        self.model = MarianMTModel.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            cache_dir=cache_dir,
        )
        
        if self.device == "cuda":
            self.model.to(self.device)
        
        self.model.eval()
        print("MarianMT model loaded successfully")
    
    def translate(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
        max_length: int = 512,
        num_beams: int = 4,
        **kwargs
    ) -> TranslationResult:
        """Translate text using MarianMT"""
        start_time = time.time()
        
        # MarianMT uses format: Helsinki-NLP/opus-mt-{src}-{tgt}
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                **kwargs
            )
        
        translated_text = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )[0]
        
        latency_ms = (time.time() - start_time) * 1000
        
        return TranslationResult(
            translated_text=translated_text,
            source_language=src_lang,
            target_language=tgt_lang,
            confidence_score=0.92,
            latency_ms=latency_ms,
            model_used="MarianMT",
            tokens_processed=len(generated_tokens[0]),
        )
    
    def translate_streaming(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
        chunk_size: int = 50,
        max_length: int = 512,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream translation output"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_ids = []
            
            for i in range(max_length):
                outputs = self.model(**inputs)
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1)
                
                generated_ids.append(next_token.item())
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                # Update inputs for next iteration
                inputs = {
                    "input_ids": torch.tensor([generated_ids], device=self.device)
                }
                
                if len(generated_ids) % chunk_size == 0:
                    partial_text = self.tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=True
                    )
                    yield partial_text
        
        final_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )
        yield final_text


class M2M100Backend(BaseModelBackend):
    """Facebook M2M100 backend for many-to-many translation"""
    
    def load_model(self, load_in_8bit: bool = False, cache_dir: Optional[str] = None):
        """Load M2M100 model"""
        print(f"Loading M2M100 model: {self.model_name} on {self.device}")
        
        dtype = torch.float16 if self.precision == "fp16" else torch.float32
        
        self.tokenizer = M2M100Tokenizer.from_pretrained(
            self.model_name,
            cache_dir=cache_dir,
        )
        
        self.model = M2M100ForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            load_in_8bit=load_in_8bit,
            cache_dir=cache_dir,
            device_map="auto" if self.device == "cuda" else None,
        )
        
        if not load_in_8bit and self.device == "cuda":
            self.model.to(self.device)
        
        self.model.eval()
        print("M2M100 model loaded successfully")
    
    def translate(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
        max_length: int = 512,
        num_beams: int = 4,
        **kwargs
    ) -> TranslationResult:
        """Translate text using M2M100"""
        start_time = time.time()
        
        src_code = self.get_language_code(src_lang, "m2m100")
        tgt_code = self.get_language_code(tgt_lang, "m2m100")
        
        self.tokenizer.src_lang = src_code
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.get_lang_id(tgt_code),
                max_length=max_length,
                num_beams=num_beams,
                **kwargs
            )
        
        translated_text = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )[0]
        
        latency_ms = (time.time() - start_time) * 1000
        
        return TranslationResult(
            translated_text=translated_text,
            source_language=src_lang,
            target_language=tgt_lang,
            confidence_score=0.93,
            latency_ms=latency_ms,
            model_used="M2M100",
            tokens_processed=len(generated_tokens[0]),
        )
    
    def translate_streaming(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
        chunk_size: int = 50,
        max_length: int = 512,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream translation output"""
        src_code = self.get_language_code(src_lang, "m2m100")
        tgt_code = self.get_language_code(tgt_lang, "m2m100")
        
        self.tokenizer.src_lang = src_code
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_ids = []
            past_key_values = None
            
            for i in range(max_length):
                if past_key_values is not None:
                    outputs = self.model(
                        input_ids=torch.tensor([[generated_ids[-1]]], device=self.device),
                        past_key_values=past_key_values,
                    )
                else:
                    outputs = self.model(**inputs)
                
                next_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1)
                
                generated_ids.append(next_token.item())
                past_key_values = outputs.past_key_values
                
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                
                if len(generated_ids) % chunk_size == 0:
                    partial_text = self.tokenizer.decode(
                        generated_ids,
                        skip_special_tokens=True
                    )
                    yield partial_text
        
        final_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )
        yield final_text


class LLMTranslationEngine:
    """
    Main LLM Translation Engine
    Manages multiple backends and provides unified interface
    """
    
    def __init__(
        self,
        backend: str = "nllb",
        model_name: Optional[str] = None,
        device: str = "auto",
        precision: str = "fp16",
        cache_dir: Optional[str] = None,
        load_in_8bit: bool = False,
    ):
        self.backend_name = backend.lower()
        self.device = self._detect_device(device)
        self.precision = precision
        self.cache_dir = cache_dir
        self.load_in_8bit = load_in_8bit
        
        # Select default model if not specified
        if model_name is None:
            default_models = {
                "nllb": "facebook/nllb-200-distilled-600M",
                "marian": "Helsinki-NLP/opus-mt-en-es",  # Example pair
                "m2m100": "facebook/m2m100_418M",
            }
            model_name = default_models.get(backend, "facebook/nllb-200-distilled-600M")
        
        self.model_name = model_name
        self.backend: Optional[BaseModelBackend] = None
        self._initialized = False
        self._lock = threading.Lock()
    
    def _detect_device(self, device: str) -> str:
        """Auto-detect best available device"""
        if device != "auto":
            return device
        
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    def initialize(self):
        """Initialize the translation engine"""
        with self._lock:
            if self._initialized:
                return
            
            print(f"Initializing {self.backend_name} backend on {self.device}")
            
            if self.backend_name == "nllb":
                self.backend = NLLBBackend(
                    self.model_name,
                    self.device,
                    self.precision
                )
                self.backend.load_model(
                    load_in_8bit=self.load_in_8bit,
                    cache_dir=self.cache_dir
                )
            elif self.backend_name == "marian":
                self.backend = MarianMTBackend(
                    self.model_name,
                    self.device,
                    self.precision
                )
                self.backend.load_model(cache_dir=self.cache_dir)
            elif self.backend_name == "m2m100":
                self.backend = M2M100Backend(
                    self.model_name,
                    self.device,
                    self.precision
                )
                self.backend.load_model(
                    load_in_8bit=self.load_in_8bit,
                    cache_dir=self.cache_dir
                )
            else:
                raise ValueError(f"Unsupported backend: {self.backend_name}")
            
            self._initialized = True
            print(f"{self.backend_name} engine initialized successfully")
    
    def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
        max_length: int = 512,
        num_beams: int = 4,
        temperature: float = 1.0,
        **kwargs
    ) -> TranslationResult:
        """
        Translate text from source to target language
        
        Args:
            text: Input text to translate
            source_language: Source language code (e.g., 'en', 'es')
            target_language: Target language code
            max_length: Maximum output length
            num_beams: Beam search size
            temperature: Sampling temperature
        
        Returns:
            TranslationResult with translated text and metadata
        """
        if not self._initialized:
            self.initialize()
        
        return self.backend.translate(
            text=text,
            src_lang=source_language,
            tgt_lang=target_language,
            max_length=max_length,
            num_beams=num_beams,
            temperature=temperature,
            **kwargs
        )
    
    def translate_streaming(
        self,
        text: str,
        source_language: str,
        target_language: str,
        chunk_size: int = 50,
        max_length: int = 512,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        Stream translation output token by token
        
        Args:
            text: Input text to translate
            source_language: Source language code
            target_language: Target language code
            chunk_size: Number of tokens per chunk
            max_length: Maximum output length
        
        Yields:
            Partial translation strings
        """
        if not self._initialized:
            self.initialize()
        
        yield from self.backend.translate_streaming(
            text=text,
            src_lang=source_language,
            tgt_lang=target_language,
            chunk_size=chunk_size,
            max_length=max_length,
            **kwargs
        )
    
    def translate_batch(
        self,
        texts: List[str],
        source_language: str,
        target_language: str,
        max_length: int = 512,
        num_beams: int = 4,
        **kwargs
    ) -> List[TranslationResult]:
        """
        Translate multiple texts in batch
        
        Args:
            texts: List of texts to translate
            source_language: Source language code
            target_language: Target language code
            max_length: Maximum output length
            num_beams: Beam search size
        
        Returns:
            List of TranslationResult objects
        """
        if not self._initialized:
            self.initialize()
        
        results = []
        for text in texts:
            result = self.backend.translate(
                text=text,
                src_lang=source_language,
                tgt_lang=target_language,
                max_length=max_length,
                num_beams=num_beams,
                **kwargs
            )
            results.append(result)
        
        return results
    
    def unload(self):
        """Unload model from memory"""
        if self.backend and self.backend.model:
            del self.backend.model
            del self.backend.tokenizer
            self.backend = None
            self._initialized = False
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print("Model unloaded from memory")
