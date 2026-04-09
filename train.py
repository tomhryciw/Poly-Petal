import torch
import torch.nn as nn
import torchaudio
import numpy as np
import json
import os
from torch.utils.data import Dataset, DataLoader

# ─── Dataset ───────────────────────────────────────────────────────────────────

class GuitarDataset(Dataset):
    def __init__(self, labels_file, processed_folder):
        with open(labels_file) as f:
            self.labels = json.load(f)
        self.processed_folder = processed_folder
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=44100,
            n_mels=64,
            n_fft=1024,
            hop_length=512
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = self.labels[idx]
        
        # load audio
        path = os.path.join(self.processed_folder, item['file'])
        waveform, sr = torchaudio.load(path)
        
        # make mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # trim or pad to 5 seconds
        target_len = 44100 * 5
        if waveform.shape[1] > target_len:
            waveform = waveform[:, :target_len]
        else:
            pad = target_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        
        # convert to mel spectrogram
        mel = self.mel(waveform)
        mel = torch.log(mel + 1e-9)
        
        # get labels as tensor, normalized to 0-1
        params = item['params']
        label = torch.tensor([
            (params['gain_db'] + 6) / 18,
            params['reverb'] / 0.8,
            (params['low_shelf'] + 10) / 20,
            (params['high_shelf'] + 10) / 20,
            (params['compression'] - 1) / 9,
        ], dtype=torch.float32)
        
        return mel, label

# ─── Model ─────────────────────────────────────────────────────────────────────

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

# ─── Training ──────────────────────────────────────────────────────────────────

dataset = GuitarDataset('data/labels.json', 'data/processed')
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

model = ToneEncoder()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

print(f'Training on {len(dataset)} samples...')

for epoch in range(50):
    total_loss = 0
    for mel, label in dataloader:
        optimizer.zero_grad()
        output = model(mel)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f'Epoch {epoch+1}/50 - Loss: {avg_loss:.4f}')

# save the model
torch.save(model.state_dict(), 'tone_encoder.pth')
print('Model saved to tone_encoder.pth')