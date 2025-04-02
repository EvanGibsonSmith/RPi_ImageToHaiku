import torchaudio
import torch

# Check if a backend is set
print("Available backends:", torchaudio.list_audio_backends())
print("Current backend:", torchaudio.get_audio_backend())

# Generate a simple test waveform (1 second of silence at 22.05 kHz)
waveform = torch.zeros(1, 22050)  
torchaudio.save("test.wav", waveform, 22050)
print("Saved test.wav successfully!")
