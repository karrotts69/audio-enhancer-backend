from flask import Flask, request, send_file, jsonify, make_response
from flask_cors import CORS
import os
import tempfile
import librosa
import soundfile as sf
import numpy as np
from pydub import AudioSegment
import logging
import subprocess

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "https://karrotts69.github.io"}})

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def check_ffmpeg():
    try:
        subprocess.run(['ffmpeg', '-version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info("FFmpeg is available")
    except Exception as e:
        logger.error(f"FFmpeg check failed: {e}")

@app.route('/api/process', methods=['POST'])
def process_audio():
    try:
        logger.info("Processing started")
        check_ffmpeg()
        if 'file' not in request.files:
            logger.error("No file part in request")
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        drum_style = request.form.get('drumStyle', 'rock')
        bass_style = request.form.get('bassStyle', 'fingered')
        drum_intensity = float(request.form.get('drumIntensity', '50')) / 100
        bass_intensity = float(request.form.get('bassIntensity', '50')) / 100

        if not file.filename.endswith(('.mp3', '.wav')):
            logger.error(f"Invalid file type: {file.filename}")
            return jsonify({"error": "Only MP3 and WAV supported"}), 400

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            input_path = tmp.name
            logger.info(f"Saving file to {input_path}")
            file.save(input_path)

        logger.info("Loading audio")
        try:
            y, sr = librosa.load(input_path, sr=None)
            logger.info(f"Audio loaded: length={len(y)}, sample rate={sr}")
        except librosa.LibrosaError as e:
            logger.warning(f"Librosa failed: {e}. Falling back to pydub")
            audio = AudioSegment.from_file(input_path)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as wav_tmp:
                wav_path = wav_tmp.name
                audio.export(wav_path, format="wav")
                y, sr = librosa.load(wav_path, sr=None)
                logger.info(f"Fallback loaded: length={len(y)}, sample rate={sr}")
                os.remove(wav_path)

        logger.info("Processing audio")
        t = np.linspace(0, len(y) / sr, len(y))
        bass_freq = 60 if bass_style == "fingered" else 40
        bass = bass_intensity * np.sin(2 * np.pi * bass_freq * t)
        drums = drum_intensity * np.random.normal(0, 0.1, len(y)) if drum_style == "rock" else \
                drum_intensity * np.random.normal(0, 0.05, len(y))
        processed_y = y + bass + drums

        try:
            tempo, _ = librosa.beat.tempo(y=y, sr=sr)
            logger.info(f"Tempo calculated: {tempo}")
        except Exception as e:
            logger.error(f"Tempo calculation failed: {e}")
            tempo = 120  # Fallback

        key = "C Major"  # Placeholder

        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_wav:
            temp_wav_path = tmp_wav.name
            logger.info(f"Writing WAV to {temp_wav_path}")
            sf.write(temp_wav_path, processed_y, sr)

            if file.filename.endswith('.mp3'):
                logger.info("Converting to MP3")
                wav_audio = AudioSegment.from_wav(temp_wav_path)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_mp3:
                    output_path = tmp_mp3.name
                    wav_audio.export(output_path, format='mp3')
                    mime_type = 'audio/mpeg'
                    os.remove(temp_wav_path)
            else:
                output_path = temp_wav_path
                mime_type = 'audio/wav'

        os.remove(input_path)
        logger.info(f"Sending file: {output_path}")
        response = make_response(send_file(output_path, mimetype=mime_type))
        response.headers['X-Tempo'] = str(tempo)
        response.headers['X-Key'] = key
        os.remove(output_path)
        return response

    except librosa.LibrosaError as e:
        logger.error(f"Audio processing failed: {e}")
        return jsonify({"error": "Invalid audio file"}), 400
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)