import sys
import os
import demucs.separate
from pipeline import predict_params

def test(mp3_path):
    print(f'\n🎵 Processing: {mp3_path}')
    
    print('\n[1/2] Running Demucs separation...')
    demucs.separate.main([mp3_path])
    
    songname = os.path.splitext(os.path.basename(mp3_path))[0]
    guitar_stem = os.path.join('separated', 'htdemucs', songname, 'other.wav')
    
    if not os.path.exists(guitar_stem):
        print('❌ Separation failed')
        return
    
    print(f' Guitar stem saved to: {guitar_stem}')
    
    print('\n[2/2] Running tone encoder...')
    params = predict_params(guitar_stem)
    
    print('\n Predicted tone parameters:')
    print(f'  Gain:        {params["gain_db"]:+.1f} dB  {"█" * int((params["gain_db"]+6)/2)}')
    print(f'  Reverb:      {params["reverb"]:.2f}      {"█" * int(params["reverb"]*20)}')
    print(f'  Low shelf:   {params["low_shelf"]:+.1f} dB  {"█" * int((params["low_shelf"]+10)/2)}')
    print(f'  High shelf:  {params["high_shelf"]:+.1f} dB  {"█" * int((params["high_shelf"]+10)/2)}')
    print(f'  Compression: {params["compression"]:.1f}x     {"█" * int(params["compression"]/1.5)}')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python test_pipeline.py <path_to_mp3>')
    else:
        test(sys.argv[1])