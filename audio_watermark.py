import numpy as np 
from scipy.io import wavfile 
from scipy import signal 
from pydub import AudioSegment

def generate_noise(length, seed=42):
    np.random.seed(seed)
    return np.random.choice([-1, 1], size=length)

def embed_watermark(audio_data, watermark, embedding_strength=0.05):
    watermark_extended = np.tile(watermark, int(np.ceil(len(audio_data) / len(watermark))))
    watermark_extended = watermark_extended[:len(audio_data)]

    # Embed the watermark by adding a scaled version of it to the audio data 
    watermarked_audio = audio_data + embedding_strength * watermark_extended 

    return np.clip(watermarked_audio, -1.0, 1.0)

def save_watermarked_audio(filename, fs, watermarked_audio):
    wavfile.write(filename, fs, watermarked_audio)

def extract_watermark(audio_data, watermark_length):
    pass


# Load the audio 
audio_segment = AudioSegment.from_file("audio.wav") 
samples = np.array(audio_segment.get_array_of_samples())

# Normalize audio samples 
audio_data = samples / (2**15)

watermark = generate_noise(len(audio_data)) 

watermarked_audio = embed_watermark(audio_data, watermark)

# Scale watermarked audio back to original amplitutude range and save
watermarked_audio_scaled = (watermarked_audio * (2**15)).astype(np.int16)
save_watermarked_audio("watermarked_audio.wav", audio_segment.frame_rate, watermarked_audio_scaled)