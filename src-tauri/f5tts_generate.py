#!/usr/bin/env python3
"""
F5-TTS voice generation script with advanced controls
Generates speech from text using F5-TTS model with prosody, post-processing, and presets
"""

import sys
import json
import tempfile
import os
from pathlib import Path

try:
    from f5_tts.api import F5TTS  # type: ignore
except ImportError:
    F5TTS = None  # type: ignore


def cleanup_torch():
    """Aggressively release GPU/CPU memory after synthesis."""
    try:
        import torch

        # Release CUDA/ROCm caches if applicable
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        # Release MPS memory on Apple (no-op on Linux)
        if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()  # type: ignore[attr-defined]
    except Exception:
        # Best-effort cleanup; ignore failures
        pass

def apply_post_processing(audio_path, settings):
    """
    Apply post-processing effects to audio
    - EQ (warmth, presence, air)
    - Reverb (room size, decay)
    - Humanization (breath, de-esser)
    - Normalization
    """
    try:
        from pydub import AudioSegment
        from pydub.effects import normalize, compress_dynamic_range
        
        audio = AudioSegment.from_file(audio_path)
        original_loudness = audio.dBFS
        
        # Apply warmth (low-end boost/cut)
        warmth = settings.get('warmth', 0.0)  # -1.0 to 1.0
        if abs(warmth) > 0.01:
            # Boost/cut low frequencies (100-500 Hz)
            db_change = warmth * 6  # Up to ±6dB
            audio = audio.low_pass_filter(500).apply_gain(db_change) + audio.high_pass_filter(500)
        
        # Apply presence boost (2-5 kHz range)
        presence = settings.get('presence', 0.0)  # 0.0 to 1.0
        if presence > 0.01:
            # Boost mid-high frequencies for clarity
            db_change = presence * 4  # Up to +4dB
            high_shelf = audio.high_pass_filter(2000)
            audio = audio + high_shelf.apply_gain(db_change)
        
        # Apply air/brightness (8+ kHz)
        air = settings.get('air', 0.0)  # 0.0 to 1.0
        if air > 0.01:
            # Boost high frequencies for airiness
            db_change = air * 5  # Up to +5dB
            airy = audio.high_pass_filter(8000).apply_gain(db_change)
            audio = audio.overlay(airy)
        
        # Apply reverb (room size and decay)
        reverb_size = settings.get('reverb_size', 0.0)  # 0.0 to 1.0
        reverb_decay = settings.get('reverb_decay', 0.5)  # 0.0 to 1.0
        if reverb_size > 0.01:
            # Create reverb tail with delay and decay
            delay_ms = int(reverb_size * 150)  # 0-150ms initial delay
            decay_db = -(1.0 - reverb_decay) * 30  # 0 to -30dB decay
            reverb_tail = audio.apply_gain(decay_db)
            # Add reverb as overlay with delay
            audio = audio.overlay(reverb_tail, position=delay_ms)
        
        # Breath volume (simulated with gentle noise)
        breath_volume = settings.get('breath_volume', 0.0)  # 0.0 to 1.0
        if breath_volume > 0.01:
            # Subtle high-frequency noise for breath presence
            breath_db = -40 + (breath_volume * 20)  # -40dB to -20dB
            # Note: Actual breath insertion would require more complex processing
            # This is a placeholder that slightly increases high-freq noise floor
        
        # De-esser (reduce harsh sibilance 6-8 kHz)
        de_esser = settings.get('de_esser', 0.5)  # 0.0 to 1.0
        if de_esser > 0.01:
            # Compress harsh frequencies
            sibilant = audio.high_pass_filter(6000).low_pass_filter(8000)
            reduction_db = -de_esser * 6  # Up to -6dB reduction
            sibilant = sibilant.apply_gain(reduction_db)
            other_freqs = audio.low_pass_filter(6000) + audio.high_pass_filter(8000)
            audio = other_freqs.overlay(sibilant)
        
        # Normalize to target LUFS (maintain consistent loudness)
        if settings.get('normalize', True):
            audio = normalize(audio, headroom=0.1)
        
        # Apply limiter (prevent clipping)
        if settings.get('limiter', True):
            audio = compress_dynamic_range(audio, threshold=-20.0, ratio=4.0, attack=5.0, release=50.0)
        
        # Save processed audio
        audio.export(audio_path, format="mp3", bitrate="192k")
        return True
        
    except ImportError:
        # If pydub not available, skip post-processing
        print("Warning: pydub not available, skipping post-processing", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Warning: Post-processing failed: {e}", file=sys.stderr)
        return False

def generate_speech(text, voice="female_neutral", speed=1.0, pitch=1.0, 
                   prosody=0.5, stability=0.7, settings=None):
    """
    Generate speech using F5-TTS with advanced controls
    
    Args:
        text: Text to convert to speech
        voice: Voice profile (speaker embedding)
        speed: Speech speed (0.7-1.3)
        pitch: Pitch adjustment (0.85-1.15)
        prosody: Expressiveness (0.0-1.0)
        stability: Voice stability (0.0-1.0)
        settings: Dict with additional settings (pauses, breath, post-fx)
    """
    if settings is None:
        settings = {}
    
    try:
        # Try to import F5-TTS
        if F5TTS is None or os.environ.get("TTS_FORCE_EDGE", "0") == "1":
            # Fallback to edge-tts with enhanced processing
            import edge_tts
            import asyncio
            
            async def _generate():
                # Map voice profiles to edge-tts voices
                voice_map = {
                    'male_neutral': 'en-US-GuyNeural',
                    'male_deep': 'en-US-EricNeural',
                    'male_soft': 'en-US-AndrewNeural',
                    'female_neutral': 'en-US-JennyNeural',
                    'female_warm': 'en-US-AriaNeural',
                    'female_bright': 'en-US-MichelleNeural',
                    'narrator': 'en-US-AriaNeural',  # switch narrator to female voice
                    'androgynous': 'en-US-AvaNeural',
                }
                
                voice_id = voice_map.get(voice, 'en-US-JennyNeural')
                
                # Calculate rate from speed and prosody
                rate_percent = int((speed - 1) * 100)
                
                # Adjust pitch
                pitch_percent = int((pitch - 1) * 50)
                
                # Create communicate object
                communicate = edge_tts.Communicate(
                    text, 
                    voice_id,
                    rate=f"{rate_percent:+d}%",
                    pitch=f"{pitch_percent:+d}Hz"
                )
                
                # Generate temp file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                output_path = temp_file.name
                temp_file.close()
                
                await communicate.save(output_path)
                return output_path
            
            output_file = asyncio.run(_generate())
            
            # Apply post-processing with ALL settings
            post_settings = {
                'warmth': settings.get('warmth', 0.0),
                'presence': settings.get('presence', 0.0),
                'air': settings.get('air', 0.0),
                'reverb_size': settings.get('reverb_size', 0.0),
                'reverb_decay': settings.get('reverb_decay', 0.5),
                'breath_volume': settings.get('breath_volume', 0.3),
                'de_esser': settings.get('de_esser', 0.5),
                'normalize': settings.get('normalize', True),
                'limiter': settings.get('limiter', True),
            }
            
            apply_post_processing(output_file, post_settings)
            
            # Read file
            with open(output_file, 'rb') as f:
                audio_data = f.read()
            
            print(json.dumps({
                "status": "success",
                "audio_path": output_file,
                "engine": "edge-tts"
            }))
            return
        
        # Use F5-TTS if available
        model = F5TTS()
        
        # Generate with advanced settings
        audio = model.synthesize(
            text=text,
            voice=voice,
            speed=speed,
            pitch=pitch,
            prosody=prosody,
            stability=stability,
            **settings
        )
        
        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        output_path = temp_file.name
        temp_file.close()
        
        # Save audio
        import torchaudio
        torchaudio.save(output_path, audio, 24000)
        
        # Apply post-processing
        apply_post_processing(output_path, settings)
        
        print(json.dumps({"status": "success", "audio_path": output_path, "engine": "f5-tts"}))
        
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}), file=sys.stderr)
        sys.exit(1)
    finally:
        # Free as much memory as possible when the process exits
        cleanup_torch()
        if 'model' in locals():
            del model
        if 'audio' in locals():
            del audio

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"status": "error", "message": "No input provided"}), file=sys.stderr)
        sys.exit(1)
    
    try:
        args = json.loads(sys.argv[1])
        text = args.get("text", "")
        voice = args.get("voice", "female_neutral")
        speed = args.get("speed", 1.0)
        pitch = args.get("pitch", 1.0)
        prosody = args.get("prosody", 0.5)
        stability = args.get("stability", 0.7)
        
        # Parse settings - it might be a JSON string or dict
        settings_raw = args.get("settings", {})
        if isinstance(settings_raw, str):
            settings = json.loads(settings_raw)
        else:
            settings = settings_raw
        
        generate_speech(text, voice, speed, pitch, prosody, stability, settings)
    except json.JSONDecodeError as e:
        print(json.dumps({"status": "error", "message": f"Invalid JSON: {e}"}), file=sys.stderr)
        sys.exit(1)
