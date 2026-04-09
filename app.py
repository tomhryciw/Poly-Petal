from flask import Flask, request, jsonify, render_template
import os
import demucs.separate
from pipeline import predict_params

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'no file'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'no file selected'}), 400
    
    # save the uploaded mp3
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # run demucs
    demucs.separate.main([filepath])

    # get guitar stem path
    songname = os.path.splitext(file.filename)[0]
    guitar_stem = os.path.join('separated', 'htdemucs', songname, 'other.wav')

    if not os.path.exists(guitar_stem):
        return jsonify({'error': 'separation failed'}), 500

    # run neural network on guitar stem
    params = predict_params(guitar_stem)

    return jsonify({'status': 'done', 'params': params})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)