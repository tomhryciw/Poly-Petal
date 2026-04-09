import torch
import torchaudio
import torch.nn as nn
import soundfile as sf
import numpy as np
import os

class ToneEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 5),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# load model once when the module is imported
model = ToneEncoder()
model.load_state_dict(torch.load('tone_encoder.pth', map_location='cpu'))
model.eval()

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=44100,
    n_mels=64,
    n_fft=1024,
    hop_length=512
)

def predict_params(wav_path):
    waveform, sr = torchaudio.load(wav_path)
    
    # mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # trim or pad to 5 seconds
    target_len = 44100 * 5
    if waveform.shape[1] > target_len:
        waveform = waveform[:, :target_len]
    else:
        pad = target_len - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    
    # mel spectrogram
    mel = mel_transform(waveform)
    mel = torch.log(mel + 1e-9)
    mel = mel.unsqueeze(0)  # add batch dimension
    
    # run through model
    with torch.no_grad():
        output = model(mel).squeeze().numpy()
    
    # denormalize back to real values
    params = {
        'gain_db':     round(float(output[0]) * 18 - 6, 2),
        'reverb':      round(float(output[1]) * 0.8, 2),
        'low_shelf':   round(float(output[2]) * 20 - 10, 2),
        'high_shelf':  round(float(output[3]) * 20 - 10, 2),
        'compression': round(float(output[4]) * 9 + 1, 2),
    }
    
    return params