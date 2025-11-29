#!/usr/bin/env python3
"""Compare streaming vs non-streaming TTS performance."""

import dotenv
dotenv.load_dotenv()

import time
import warnings
warnings.filterwarnings('ignore')

from tts.tts import TTS
from core.config_manager import config

# Test sentences of varying lengths
TEST_CASES = [
    ("Short", "Hello, how are you?"),
    ("Medium", "The quick brown fox jumps over the lazy dog. This is a medium length sentence to test performance."),
    ("Long", "Artificial intelligence is transforming how we interact with technology. From voice assistants to autonomous vehicles, AI systems are becoming increasingly sophisticated. This longer text helps us understand how TTS performance scales with content length."),
]

def test_non_streaming(tts, text, device):
    """Test non-streaming TTS."""
    start = time.time()
    tts.play_with_device(text, device=device)
    return time.time() - start

def test_streaming(tts, text, device):
    """Test streaming TTS."""
    def text_generator():
        # Simulate chunked text like LLM would produce
        words = text.split()
        chunk = ""
        for word in words:
            chunk += word + " "
            if len(chunk) > 20:  # Yield every ~20 chars
                yield chunk
                chunk = ""
        if chunk:
            yield chunk
    
    start = time.time()
    tts.play_streaming(text_generator(), device=device)
    return time.time() - start

def main():
    tts_config = config.get_section('TTS')
    device = tts_config.get('device')
    
    print("=" * 60)
    print("TTS SPEED COMPARISON: Streaming vs Non-Streaming")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Voice: {tts_config.get('voice')}")
    print(f"Model: {tts_config.get('model')}")
    print()
    
    tts = TTS()
    
    results = []
    
    for name, text in TEST_CASES:
        print(f"Testing '{name}' ({len(text)} chars)...")
        print(f"  Text: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        # Non-streaming
        print("  [Non-streaming] Playing...", end=" ", flush=True)
        ns_time = test_non_streaming(tts, text, device)
        print(f"{ns_time:.2f}s")
        
        # Brief pause between tests
        time.sleep(0.5)
        
        # Streaming
        print("  [Streaming]     Playing...", end=" ", flush=True)
        s_time = test_streaming(tts, text, device)
        print(f"{s_time:.2f}s")
        
        # Calculate difference
        diff = ns_time - s_time
        winner = "Streaming" if s_time < ns_time else "Non-streaming"
        pct = abs(diff) / max(ns_time, s_time) * 100
        
        results.append({
            'name': name,
            'chars': len(text),
            'non_streaming': ns_time,
            'streaming': s_time,
            'winner': winner,
            'diff': abs(diff),
            'pct': pct
        })
        
        print(f"  → {winner} wins by {abs(diff):.2f}s ({pct:.0f}% faster)")
        print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Test':<10} {'Chars':<8} {'Non-Stream':<12} {'Stream':<12} {'Winner':<15}")
    print("-" * 60)
    
    for r in results:
        print(f"{r['name']:<10} {r['chars']:<8} {r['non_streaming']:.2f}s{'':<7} {r['streaming']:.2f}s{'':<7} {r['winner']}")
    
    print()
    
    # Recommendation
    stream_wins = sum(1 for r in results if r['winner'] == 'Streaming')
    if stream_wins >= len(results) / 2:
        print("RECOMMENDATION: Use streaming mode (streaming_enabled: true)")
        print("  + Faster overall performance")
        print("  + Lower perceived latency (audio starts sooner)")
    else:
        print("RECOMMENDATION: Use non-streaming mode (streaming_enabled: false)")
        print("  + More reliable on your system")
        print("  + Simpler audio pipeline")
    
    print()
    print("To change mode, edit config.yml:")
    print("  TTS:")
    print("    streaming_enabled: true  # or false")

if __name__ == "__main__":
    main()

