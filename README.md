# Real-Time Translation System

A high-performance, real-time voice and text translation system supporting 100+ languages with low-latency optimization.

## Features

- **Multi-language Support**: Translate between 100+ languages including English, Spanish, French, German, Chinese, Japanese, Korean, Arabic, Hindi, and more
- **Text-to-Text Translation**: Fast batch and streaming text translation
- **Voice-to-Voice Translation**: Complete speech-to-speech translation pipeline
- **Real-time Streaming**: Optimized for live audio stream translation
- **GPU Acceleration**: Supports CUDA for faster inference
- **Multiple Model Backends**: Choose from NLLB, MarianMT, or M2M100 models

## Installation

### Basic Installation

```bash
pip install transformers torch sentencepiece
```

### For Voice Translation

```bash
# Speech-to-Text (Whisper)
pip install openai-whisper

# Text-to-Speech (choose one or more)
pip install gTTS
pip install pyttsx3
pip install TTS  # Coqui TTS

# Audio processing
pip install soundfile numpy
```

### For GPU Acceleration (CUDA)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### Text Translation

```python
from translator import create_translator

# Create a text translator (English to Spanish)
translator = create_translator("en", "es", mode="text")

# Translate text
text = "Hello, how are you today?"
result = translator.translate(text)
print(f"Translation: {result}")

# Batch translation
texts = ["Good morning", "Thank you", "How much does this cost?"]
results = translator.translate_batch(texts)
for original, translated in zip(texts, results):
    print(f"{original} -> {translated}")
```

### Voice Translation

```python
from translator import create_translator

# Create a voice translator (English to French)
translator = create_translator("en", "fr", mode="voice")

# Load required models
translator.load_speech_to_text(model_name="base")  # Options: tiny, base, small, medium, large
translator.load_text_to_speech(provider="google")  # Options: google, pyttsx3, coqui

# Translate voice to voice
result = translator.translate_voice("input.wav", "output.mp3")

print(f"Original: {result['original_text']}")
print(f"Translated: {result['translated_text']}")
print(f"Audio saved to: {result['output_audio']}")
```

### Real-time Streaming

```python
import asyncio
from translator import create_translator

async def main():
    # Create real-time translator
    translator = create_translator("en", "ja", mode="realtime")
    
    # Your audio stream generator (example)
    async def audio_stream():
        # Replace with your actual audio source
        # e.g., microphone input, WebSocket stream, etc.
        for chunk in get_audio_chunks():
            yield chunk
    
    # Process stream
    async for result in translator.translate_audio_stream(audio_stream()):
        if result["status"] == "partial":
            print(f"Partial: {result['translated_text']}")
        else:
            print(f"Complete: {result['translated_text']}")

# Run the async translator
# asyncio.run(main())
```

## Supported Languages

The system supports 100+ languages including:

| Language | Code | Language | Code |
|----------|------|----------|------|
| English | en | Spanish | es |
| French | fr | German | de |
| Italian | it | Portuguese | pt |
| Russian | ru | Japanese | ja |
| Korean | ko | Chinese | zh |
| Arabic | ar | Hindi | hi |
| Dutch | nl | Polish | pl |
| Turkish | tr | Vietnamese | vi |
| Thai | th | Indonesian | id |

And many more through the NLLB and M2M100 models.

## Configuration

Customize the translation behavior:

```python
from translator import TranslationConfig, TextTranslator

config = TranslationConfig(
    source_language="en",
    target_language="zh",
    model_type="nllb",  # Options: nllb, marian, m2m100
    device="cuda",      # Use "cpu" if no GPU available
    max_length=512,     # Maximum sequence length
    beam_size=4,        # Beam search size for better quality
    enable_streaming=True,
    cache_model=True
)

translator = TextTranslator(config)
translator.load_model()
```

## Model Options

### Translation Models

1. **NLLB (No Language Left Behind)**: Facebook's model supporting 200+ languages
   - Best for: Multi-language support, quality
   - Model: `facebook/nllb-200-distilled-600M`

2. **MarianMT**: Efficient bilingual models
   - Best for: Specific language pairs, speed
   - Model: `Helsinki-NLP/opus-mt-{src}-{tgt}`

3. **M2M100**: Many-to-many multilingual model
   - Best for: Direct multilingual translation
   - Model: `facebook/m2m100_418M`

### Speech-to-Text Models (Whisper)

- **tiny**: Fastest, lowest accuracy (~32MB)
- **base**: Good balance (~74MB)
- **small**: Better accuracy (~244MB)
- **medium**: High accuracy (~769MB)
- **large**: Best accuracy (~1.5GB)

### Text-to-Speech Providers

- **Google TTS (gTTS)**: Free, good quality, requires internet
- **pyttsx3**: Offline, fast, robotic voice
- **Coqui TTS**: High quality, neural voices, larger models

## Performance Optimization

### Tips for Faster Translation

1. **Use GPU**: Enable CUDA for 10-50x speedup
2. **FP16 Precision**: Automatically enabled on GPU
3. **Smaller Models**: Use distilled/tiny models for speed
4. **Batch Processing**: Translate multiple texts together
5. **Model Caching**: Keep models loaded for repeated use

### Latency Estimates

| Task | CPU | GPU |
|------|-----|-----|
| Text Translation (short) | 100-500ms | 10-50ms |
| Speech-to-Text (10s audio) | 2-5s | 0.5-2s |
| Text-to-Speech | 100-300ms | 50-150ms |
| Full Voice Pipeline | 3-10s | 1-3s |

## API Reference

### Classes

- `Language`: Enum of supported language codes
- `TranslationConfig`: Configuration dataclass
- `TextTranslator`: Text translation engine
- `VoiceTranslator`: Voice-to-voice translation pipeline
- `RealTimeTranslator`: Streaming real-time translation

### Key Methods

#### TextTranslator
- `load_model(model_name)`: Load translation model
- `translate(text, stream)`: Translate single text
- `translate_batch(texts)`: Translate multiple texts

#### VoiceTranslator
- `load_speech_to_text(model_name)`: Load Whisper model
- `load_text_to_speech(provider)`: Load TTS engine
- `speech_to_text(audio_path)`: Transcribe audio
- `text_to_speech(text, output_path)`: Synthesize speech
- `translate_voice(audio_path, output_path)`: Complete pipeline

#### RealTimeTranslator
- `translate_audio_stream(audio_stream)`: Async streaming translation
- `stop()`: Stop translation

## Examples

See `translator.py` for complete working examples.

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `max_length` or use smaller models
2. **Slow Performance**: Enable GPU, use smaller Whisper model
3. **Poor Quality**: Increase `beam_size`, use larger models
4. **Import Errors**: Install required dependencies

### Getting Help

Check the Hugging Face documentation for transformer models:
- [NLLB](https://huggingface.co/facebook/nllb-200-distilled-600M)
- [MarianMT](https://huggingface.co/Helsinki-NLP)
- [Whisper](https://github.com/openai/whisper)

## License

This project is provided as-is. The underlying models have their own licenses:
- NLLB: CC-BY-NC-SA
- MarianMT: Apache 2.0
- Whisper: MIT
- M2M100: MIT

## Contributing

Contributions welcome! Areas for improvement:
- Better streaming implementation
- VAD (Voice Activity Detection) integration
- More TTS providers
- Web interface
- Mobile app integration
