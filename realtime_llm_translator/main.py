#!/usr/bin/env python3
"""
Real-Time LLM Translator - Main Entry Point
Advanced multi-language translation system with voice and text support
"""

import argparse
import sys
from pathlib import Path

from realtime_llm_translator.core.config import Config, PipelineMode
from realtime_llm_translator.pipeline.translation_pipeline import TranslationPipeline
from realtime_llm_translator.api.server import TranslationAPI


def main():
    parser = argparse.ArgumentParser(
        description="Real-Time LLM Translator - Advanced translation system"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["text", "voice-to-text", "text-to-voice", "voice-to-voice", "api"],
        default="text",
        help="Translation mode"
    )
    
    parser.add_argument(
        "--source-lang",
        type=str,
        default="auto",
        help="Source language code (e.g., 'en', 'es', 'fr')"
    )
    
    parser.add_argument(
        "--target-lang",
        type=str,
        default="en",
        help="Target language code"
    )
    
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Text to translate"
    )
    
    parser.add_argument(
        "--audio-file",
        type=str,
        default=None,
        help="Path to audio file for speech translation"
    )
    
    parser.add_argument(
        "--api-host",
        type=str,
        default="0.0.0.0",
        help="API server host"
    )
    
    parser.add_argument(
        "--api-port",
        type=int,
        default=8000,
        help="API server port"
    )
    
    parser.add_argument(
        "--low-latency",
        action="store_true",
        help="Optimize for low latency"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Device to use for inference"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = Config(
        source_language=args.source_lang,
        target_language=args.target_lang,
    )
    
    if args.low_latency:
        config.optimize_for_low_latency()
    
    config.hardware.device = args.device
    
    if args.mode == "api":
        # Run API server
        print(f"Starting API server on {args.api_host}:{args.api_port}")
        
        api = TranslationAPI(
            config=config,
            mode=PipelineMode.TEXT_ONLY,
        )
        
        api.run(host=args.api_host, port=args.api_port)
    
    else:
        # Initialize pipeline
        mode_map = {
            "text": PipelineMode.TEXT_ONLY,
            "voice-to-text": PipelineMode.VOICE_TO_TEXT,
            "text-to-voice": PipelineMode.TEXT_TO_VOICE,
            "voice-to-voice": PipelineMode.VOICE_TO_VOICE,
        }
        
        pipeline = TranslationPipeline(
            config=config,
            mode=mode_map[args.mode],
        )
        
        try:
            if args.text:
                # Text translation
                print(f"Translating from {args.source_lang} to {args.target_lang}...")
                result = pipeline.translate_text(
                    text=args.text,
                    source_language=args.source_lang,
                    target_language=args.target_lang,
                )
                
                print(f"\nOriginal: {args.text}")
                print(f"Translated: {result.translated_text}")
                print(f"Latency: {result.total_latency_ms:.2f}ms")
            
            elif args.audio_file:
                # Speech translation
                import numpy as np
                import wave
                
                print(f"Processing audio file: {args.audio_file}")
                
                with wave.open(args.audio_file, 'rb') as wav:
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
                
                result = pipeline.translate_speech(
                    audio_data=audio_data.astype(np.float32),
                    source_language=args.source_lang,
                    target_language=args.target_lang,
                    output_speech=(args.mode == "voice-to-voice"),
                    sample_rate=sample_rate,
                )
                
                print(f"\nTranscription: {result.transcription.text}")
                print(f"Translation: {result.translated_text}")
                print(f"Total Latency: {result.total_latency_ms:.2f}ms")
                print(f"Stages: {', '.join(result.stages_completed)}")
            
            else:
                # Interactive mode
                print("Interactive mode - enter text to translate (or 'quit' to exit)")
                
                while True:
                    try:
                        text = input("\nEnter text: ").strip()
                        
                        if text.lower() in ['quit', 'exit', 'q']:
                            break
                        
                        if not text:
                            continue
                        
                        result = pipeline.translate_text(
                            text=text,
                            source_language=args.source_lang,
                            target_language=args.target_lang,
                        )
                        
                        print(f"Translation: {result.translated_text}")
                        print(f"Latency: {result.total_latency_ms:.2f}ms")
                    
                    except KeyboardInterrupt:
                        break
                    except Exception as e:
                        print(f"Error: {e}")
        
        finally:
            pipeline.unload()


if __name__ == "__main__":
    main()
