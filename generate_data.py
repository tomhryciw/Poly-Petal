import numpy as np
import soundfile as sf
import os
import json
from pedalboard import Pedalboard, Reverb, Gain, LowShelfFilter, HighShelfFilter, Compressor
import random

DRY_FOLDER = 'data/dry_guitar'
OUTPUT_FOLDER = 'data/processed'
LABELS_FILE = 'data/labels.json'

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def random_params():
    return {
        'gain_db':     round(random.uniform(-6, 12), 2),
        'reverb':      round(random.uniform(0, 0.8), 2),
        'low_shelf':   round(random.uniform(-10, 10), 2),
        'high_shelf':  round(random.uniform(-10, 10), 2),
        'compression': round(random.uniform(1, 10), 2),
    }

def apply_effects(audio, sr, params):
    board = Pedalboard([
        Gain(gain_db=params['gain_db']),
        LowShelfFilter(cutoff_frequency_hz=300, gain_db=params['low_shelf']),
        HighShelfFilter(cutoff_frequency_hz=3000, gain_db=params['high_shelf']),
        Compressor(threshold_db=-20, ratio=params['compression']),
        Reverb(room_size=params['reverb']),
    ])
    return board(audio, sr)

labels = []
sample_count = 0

for filename in os.listdir(DRY_FOLDER):
    if not filename.endswith('.wav'):
        continue
    
    filepath = os.path.join(DRY_FOLDER, filename)
    audio, sr = sf.read(filepath)
    
    # make mono if stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    audio = audio.astype(np.float32)

    # generate 20 variations per sample
    for i in range(200):
        params = random_params()
        processed = apply_effects(audio, sr, params)
        
        out_filename = f'sample_{sample_count:04d}.wav'
        out_path = os.path.join(OUTPUT_FOLDER, out_filename)
        sf.write(out_path, processed, sr)
        
        labels.append({
            'file': out_filename,
            'params': params
        })
        
        sample_count += 1
        print(f'Generated {sample_count} samples...')

with open(LABELS_FILE, 'w') as f:
    json.dump(labels, f, indent=2)

print(f'Done! Generated {sample_count} samples with labels saved to {LABELS_FILE}')