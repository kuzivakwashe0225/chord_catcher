import numpy as np
import sounddevice as sd
import time
from sklearn.metrics.pairwise import cosine_similarity
from chord_templates import chord_templates, chord_names  # Ensure these are correct

# Constants
SAMPLERATE = 44100
DURATION = 1.0  # seconds per sample
SIMILARITY_THRESHOLD = 0.35  # Increased threshold for better filtering
SILENCE_RMS_THRESHOLD = 0.005  # More robust silence detection using RMS energy

def record_audio(duration=1.0, samplerate=44100):
    print("🎙️ Recording...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    return audio.flatten()

def compute_pcp(audio, samplerate=44100):
    # FFT and magnitude spectrum
    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), d=1/samplerate)

    # Filter out low energy bins
    spectrum[spectrum < np.max(spectrum) * 0.1] = 0

    # Pitch class profile
    pitch_classes = np.zeros(12)
    for mag, freq in zip(spectrum, freqs):
        if freq < 20 or freq > 5000:
            continue
        midi = int(round(69 + 12 * np.log2(freq / 440.0)))
        pc = midi % 12
        pitch_classes[pc] += mag

    # Normalize
    total = np.sum(pitch_classes)
    if total > 0:
        pitch_classes /= total

    return pitch_classes

def is_silence(audio):
    rms = np.sqrt(np.mean(audio**2))
    return rms < SILENCE_RMS_THRESHOLD

def main():
    print("\n🎸 Chord Catcher - Real-Time Guitar Chord Identifier")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            audio = record_audio(DURATION, SAMPLERATE)

            if is_silence(audio):
                print("🔇 Silence Detected: Display -> Unknown\n")
                continue

            pcp = compute_pcp(audio)
            print(f"🎼 PCP: {np.round(pcp, 3)}")

            # Match chord using cosine similarity
            similarities = cosine_similarity([pcp], chord_templates)[0]
            best_match_index = np.argmax(similarities)
            best_similarity = similarities[best_match_index]
            best_chord = chord_names[best_match_index]

            print(f"🔍 Similarities: {np.round(similarities, 2)}")
            print(f"🎵 Best Match: {best_chord} (Score: {best_similarity:.2f})")

            if best_similarity >= SIMILARITY_THRESHOLD:
                print(f"✅ Detected Chord: {best_chord}\n")
            else:
                print("❌ Detected Chord: Unknown (Low Confidence)\n")

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")

if __name__ == "__main__":
    main()
