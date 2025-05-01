from flask import Flask, request, jsonify
from flask_cors import CORS  # Allow requests from your mobile app
import os
from datetime import datetime
import torch
import torchaudio
from tools import audio

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return "Welcome to the Speech Recognition API!"

@app.route('/transcribe', methods=['POST'])
def transcibe():
    print(f"Received request: {request.files}")
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    print(f"Received audio file: {audio_file}")
    if audio_file.filename == '':
        return jsonify({'error': 'No selected audio file'}), 400

    try:
        file_name = datetime.now().strftime('%I_%M_%S_%p')
        save_path = os.path.join(UPLOAD_FOLDER, file_name+'.m4a')
        audio_file.save(save_path)
        audio.to_wav(save_path, save_path.replace('.m4a', '.wav'))        
        os.remove(save_path)
        
        return jsonify({'transcript': "Hello World!"}), 200

    except Exception as e:
        raise e
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)