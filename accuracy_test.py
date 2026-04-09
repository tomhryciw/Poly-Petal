import torch
import torchaudio
import torch.nn as nn
import json
import os
import random
from pipeline import predict_params, model

def test_accuracy(num_samples=10):
    with open('data/labels.json') as f:
        labels = json.load(f)
    
    # pick random samples to test
    test_samples = random.sample(labels, num_samples)
    
    errors = {
        'gain_db': [],
        'reverb': [],
        'low_shelf': [],
        'high_shelf': [],
        'compression': []
    }
    
    print(f'\nTesting on {num_samples} samples...\n')
    print(f'{"FILE":<20} {"PARAM":<12} {"ACTUAL":>8} {"PREDICTED":>10} {"ERROR":>8}')
    print('─' * 62)
    
    for item in test_samples:
        wav_path = os.path.join('data/processed', item['file'])
        actual = item['params']
        predicted = predict_params(wav_path)
        
        for key in errors:
            err = abs(actual[key] - predicted[key])
            errors[key].append(err)
        
        # print gain and reverb as a sample for each file
        print(f'{item["file"]:<20} {"gain_db":<12} {actual["gain_db"]:>8.2f} {predicted["gain_db"]:>10.2f} {abs(actual["gain_db"]-predicted["gain_db"]):>8.2f}')
        print(f'{"":20} {"reverb":<12} {actual["reverb"]:>8.2f} {predicted["reverb"]:>10.2f} {abs(actual["reverb"]-predicted["reverb"]):>8.2f}')
        print(f'{"":20} {"low_shelf":<12} {actual["low_shelf"]:>8.2f} {predicted["low_shelf"]:>10.2f} {abs(actual["low_shelf"]-predicted["low_shelf"]):>8.2f}')
        print(f'{"":20} {"high_shelf":<12} {actual["high_shelf"]:>8.2f} {predicted["high_shelf"]:>10.2f} {abs(actual["high_shelf"]-predicted["high_shelf"]):>8.2f}')
        print(f'{"":20} {"compression":<12} {actual["compression"]:>8.2f} {predicted["compression"]:>10.2f} {abs(actual["compression"]-predicted["compression"]):>8.2f}')
        print()
    
    print('─' * 62)
    print('AVERAGE ERRORS:')
    for key, errs in errors.items():
        avg = sum(errs) / len(errs)
        bar = '█' * int(avg * 3)
        print(f'  {key:<14} {avg:.3f}  {bar}')

if __name__ == '__main__':
    test_accuracy(10)