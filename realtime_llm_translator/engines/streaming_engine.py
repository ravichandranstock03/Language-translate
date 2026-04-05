"""
Advanced Streaming Engine for Real-time Audio Processing
Handles audio buffering, voice activity detection, and low-latency streaming
"""

import numpy as np
from typing import Optional, Dict, List, Generator, Callable, Tuple, AsyncGenerator
from dataclasses import dataclass, field
import threading
from queue import Queue, Empty
import time
from collections import deque
import asyncio


@dataclass
class StreamConfig:
    """Configuration for streaming engine"""
    sample_rate: int = 16000
    chunk_size_ms: int = 100  # Milliseconds per chunk
    buffer_size_seconds: float = 3.0  # Total buffer size
    overlap_ms: int = 50  # Overlap between chunks
    silence_threshold: float = 0.01  # VAD threshold
    min_speech_duration_ms: int = 200  # Minimum speech to process
    max_silence_duration_ms: int = 1000  # Max silence before flush
    
    @property
    def chunk_size_samples(self) -> int:
        return int(self.sample_rate * self.chunk_size_ms / 1000)
    
    @property
    def overlap_samples(self) -> int:
        return int(self.sample_rate * self.overlap_ms / 1000)
    
    @property
    def buffer_size_samples(self) -> int:
        return int(self.sample_rate * self.buffer_size_seconds)


@dataclass
class AudioChunk:
    """Container for audio chunk with metadata"""
    data: np.ndarray
    timestamp: float
    is_speech: bool = True
    confidence: float = 1.0
    sequence_number: int = 0


class VoiceActivityDetector:
    """Simple energy-based voice activity detection"""
    
    def __init__(self, threshold: float = 0.01, min_duration_ms: int = 200):
        self.threshold = threshold
        self.min_duration_samples = int(16000 * min_duration_ms / 1000)
        self.speech_start: Optional[int] = None
        self.is_speaking = False
    
    def detect(self, audio: np.ndarray) -> Tuple[bool, float]:
        """
        Detect if audio contains speech
        
        Returns:
            Tuple of (is_speech, confidence)
        """
        # Calculate RMS energy
        energy = np.sqrt(np.mean(audio ** 2))
        
        is_speech = energy > self.threshold
        confidence = min(1.0, energy / (self.threshold * 2))
        
        return is_speech, confidence
    
    def process_stream(self, audio: np.ndarray) -> List[Tuple[int, int]]:
        """
        Process continuous audio stream and return speech segments
        
        Returns:
            List of (start_sample, end_sample) tuples
        """
        segments = []
        is_speech, _ = self.detect(audio)
        
        if is_speech and not self.is_speaking:
            # Speech started
            self.speech_start = 0
            self.is_speaking = True
        elif not is_speech and self.is_speaking:
            # Speech ended
            if self.speech_start is not None:
                duration = len(audio) - self.speech_start
                if duration >= self.min_duration_samples:
                    segments.append((self.speech_start, len(audio)))
            self.is_speaking = False
            self.speech_start = None
        
        return segments


class CircularBuffer:
    """Thread-safe circular buffer for audio streaming"""
    
    def __init__(self, max_size: int, dtype: np.dtype = np.float32):
        self.max_size = max_size
        self.dtype = dtype
        self.buffer = np.zeros(max_size, dtype=dtype)
        self.write_pos = 0
        self.read_pos = 0
        self.lock = threading.Lock()
        self.size = 0
    
    def write(self, data: np.ndarray) -> int:
        """Write data to buffer, returns number of samples written"""
        with self.lock:
            available = self.max_size - self.size
            to_write = min(len(data), available)
            
            if to_write == 0:
                return 0
            
            # Handle wrap-around
            first_part = min(to_write, self.max_size - self.write_pos)
            self.buffer[self.write_pos:self.write_pos + first_part] = data[:first_part]
            
            if first_part < to_write:
                second_part = to_write - first_part
                self.buffer[0:second_part] = data[first_part:to_write]
            
            self.write_pos = (self.write_pos + to_write) % self.max_size
            self.size += to_write
            
            return to_write
    
    def read(self, num_samples: int) -> np.ndarray:
        """Read data from buffer"""
        with self.lock:
            to_read = min(num_samples, self.size)
            
            if to_read == 0:
                return np.array([], dtype=self.dtype)
            
            result = np.zeros(to_read, dtype=self.dtype)
            
            # Handle wrap-around
            first_part = min(to_read, self.max_size - self.read_pos)
            result[:first_part] = self.buffer[self.read_pos:self.read_pos + first_part]
            
            if first_part < to_read:
                second_part = to_read - first_part
                result[first_part:] = self.buffer[0:second_part]
            
            self.read_pos = (self.read_pos + to_read) % self.max_size
            self.size -= to_read
            
            return result
    
    def peek(self, num_samples: int, offset: int = 0) -> np.ndarray:
        """Peek at data without consuming it"""
        with self.lock:
            to_read = min(num_samples, self.size - offset)
            
            if to_read <= 0:
                return np.array([], dtype=self.dtype)
            
            result = np.zeros(to_read, dtype=self.dtype)
            start_pos = (self.read_pos + offset) % self.max_size
            
            # Handle wrap-around
            first_part = min(to_read, self.max_size - start_pos)
            result[:first_part] = self.buffer[start_pos:start_pos + first_part]
            
            if first_part < to_read:
                second_part = to_read - first_part
                result[first_part:] = self.buffer[0:second_part]
            
            return result
    
    def available(self) -> int:
        """Get number of available samples"""
        with self.lock:
            return self.size
    
    def clear(self):
        """Clear the buffer"""
        with self.lock:
            self.buffer.fill(0)
            self.write_pos = 0
            self.read_pos = 0
            self.size = 0


class StreamingEngine:
    """
    Advanced streaming engine for real-time audio processing
    Manages audio buffering, VAD, and coordinates with translation engines
    """
    
    def __init__(
        self,
        config: Optional[StreamConfig] = None,
        on_chunk_processed: Optional[Callable[[str], None]] = None,
        on_translation_ready: Optional[Callable[[str], None]] = None,
    ):
        self.config = config or StreamConfig()
        self.on_chunk_processed = on_chunk_processed
        self.on_translation_ready = on_translation_ready
        
        self.audio_buffer = CircularBuffer(self.config.buffer_size_samples)
        self.vad = VoiceActivityDetector(
            threshold=self.config.silence_threshold,
            min_duration_ms=self.config.min_speech_duration_ms
        )
        
        self.input_queue: Queue = Queue()
        self.output_queue: Queue = Queue()
        self.translation_queue: Queue = Queue()
        
        self._running = False
        self._processing_thread: Optional[threading.Thread] = None
        self._streaming_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        self.sequence_number = 0
        self.last_speech_time: float = 0
        self.pending_audio: List[np.ndarray] = []
    
    def start(self):
        """Start the streaming engine"""
        with self._lock:
            if self._running:
                return
            
            self._running = True
            self._processing_thread = threading.Thread(target=self._process_audio_loop, daemon=True)
            self._processing_thread.start()
            print("Streaming engine started")
    
    def stop(self):
        """Stop the streaming engine"""
        with self._lock:
            self._running = False
            
            if self._processing_thread:
                self._processing_thread.join(timeout=2.0)
            
            self.audio_buffer.clear()
            while not self.input_queue.empty():
                try:
                    self.input_queue.get_nowait()
                except Empty:
                    break
            
            print("Streaming engine stopped")
    
    def push_audio(self, audio_data: np.ndarray):
        """Push audio data into the streaming buffer"""
        if not self._running:
            self.start()
        
        self.input_queue.put(audio_data)
    
    def _process_audio_loop(self):
        """Main processing loop for audio streaming"""
        chunk_accumulator = []
        accumulated_samples = 0
        
        while self._running:
            try:
                # Get audio from input queue
                audio = self.input_queue.get(timeout=0.1)
                
                # Add to buffer
                self.audio_buffer.write(audio)
                
                # Accumulate for chunk processing
                chunk_accumulator.append(audio)
                accumulated_samples += len(audio)
                
                # Process when we have enough samples
                if accumulated_samples >= self.config.chunk_size_samples:
                    combined_chunk = np.concatenate(chunk_accumulator)
                    
                    # Apply overlap-add if needed
                    if len(combined_chunk) > self.config.chunk_size_samples:
                        combined_chunk = combined_chunk[:self.config.chunk_size_samples]
                    
                    # Detect voice activity
                    is_speech, confidence = self.vad.detect(combined_chunk)
                    
                    if is_speech:
                        self.last_speech_time = time.time()
                        
                        # Create audio chunk
                        chunk = AudioChunk(
                            data=combined_chunk,
                            timestamp=time.time(),
                            is_speech=is_speech,
                            confidence=confidence,
                            sequence_number=self.sequence_number
                        )
                        self.sequence_number += 1
                        
                        # Send to output
                        self.output_queue.put(chunk)
                    
                    chunk_accumulator = []
                    accumulated_samples = 0
                
                # Check for silence timeout
                if (time.time() - self.last_speech_time) * 1000 > self.config.max_silence_duration_ms:
                    if chunk_accumulator:
                        # Flush remaining audio
                        combined_chunk = np.concatenate(chunk_accumulator)
                        chunk = AudioChunk(
                            data=combined_chunk,
                            timestamp=time.time(),
                            is_speech=False,
                            confidence=0.5,
                            sequence_number=self.sequence_number
                        )
                        self.output_queue.put(chunk)
                        chunk_accumulator = []
                        accumulated_samples = 0
                    
                    self.last_speech_time = time.time()
                    
            except Empty:
                continue
            except Exception as e:
                print(f"Error in audio processing loop: {e}")
    
    def get_audio_chunks(self, timeout: float = 1.0) -> Generator[AudioChunk, None, None]:
        """Get processed audio chunks"""
        while self._running:
            try:
                chunk = self.output_queue.get(timeout=timeout)
                yield chunk
            except Empty:
                continue
    
    def process_with_callback(
        self,
        process_fn: Callable[[np.ndarray], str],
        blocking: bool = True,
    ):
        """
        Process audio chunks with a callback function
        
        Args:
            process_fn: Function that takes audio and returns text
            blocking: If True, block until translation is ready
        """
        for chunk in self.get_audio_chunks():
            if chunk.is_speech:
                # Process audio chunk
                translated_text = process_fn(chunk.data)
                
                if self.on_chunk_processed:
                    self.on_chunk_processed(translated_text)
                
                if blocking and self.on_translation_ready:
                    self.on_translation_ready(translated_text)
    
    async def process_async(
        self,
        process_fn: Callable[[np.ndarray], str],
    ) -> AsyncGenerator[str, None]:
        """
        Async generator for processing audio chunks
        
        Yields:
            Translated text strings
        """
        loop = asyncio.get_event_loop()
        
        while self._running:
            try:
                chunk = await loop.run_in_executor(None, lambda: self.output_queue.get(timeout=0.1))
                
                if chunk.is_speech:
                    # Run processing in executor to avoid blocking
                    translated_text = await loop.run_in_executor(
                        None,
                        lambda: process_fn(chunk.data)
                    )
                    
                    yield translated_text
                    
            except Empty:
                await asyncio.sleep(0.01)
            except Exception as e:
                print(f"Error in async processing: {e}")
    
    def get_buffer_state(self) -> Dict:
        """Get current buffer state"""
        return {
            "buffer_available": self.audio_buffer.available(),
            "buffer_size": self.config.buffer_size_samples,
            "input_queue_size": self.input_queue.qsize(),
            "output_queue_size": self.output_queue.qsize(),
            "is_running": self._running,
            "last_speech_time": self.last_speech_time,
        }
    
    def reset(self):
        """Reset the streaming engine state"""
        self.audio_buffer.clear()
        self.vad.is_speaking = False
        self.vad.speech_start = None
        self.sequence_number = 0
        self.pending_audio = []
        
        while not self.input_queue.empty():
            try:
                self.input_queue.get_nowait()
            except Empty:
                break
        
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except Empty:
                break
