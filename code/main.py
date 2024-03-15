import os
import time
import base64

from flask import Flask, jsonify, request
from flask_cors import CORS

from model import predict

app = Flask('voice-impersonation-backend', static_folder=os.getenv('REACT_BUILD_PATH'), static_url_path='/')

cors = CORS(app)

@app.route('/')
def index():
    return 'hello'

@app.route('/api')
def api_index():
    return 'Welcome to the voice impersonation detection api!\nWith one whole endpoint!!'

@app.route('/api/predict', methods=['POST'])
def api_predict():
    start_time = time.perf_counter()
    request_data = request.get_json()
    try:
        audio_file = request_data['sample']
        prediction = predict(audio_file)
        return jsonify({
            "status": "success",
            "analysis": prediction['analysis'],
            "responseTime": (time.perf_counter() - start_time) * 100
        }), 200
    except Exception:
        return jsonify({
            'message': 'failure',
            "analysis": {},
            "responseTime": (time.perf_counter() - start_time) * 100
        }), 500
    
@app.route('/api/predict/binary', methods=['POST']) 
def api_predict_binary():
    start_time = time.perf_counter()
    audio_file = request.get_data()
    with open('data/temp/current_file.wav', 'wb') as file:
        file.write(audio_file)
    prediction = predict('data/temp/current_file.wav')
    try:
        return jsonify({
            "status": "success",
            "analysis": prediction['analysis'],
            "responseTime": (time.perf_counter() - start_time) * 100
        }), 200
    except Exception:
        return jsonify({
            'message': 'failure',
            "analysis": {},
            "responseTime": (time.perf_counter() - start_time) * 100
        }), 500
    
@app.route('/api/predict/base64', methods=['POST'])
def api_predict_base64():
    start_time = time.perf_counter()
    request_data = request.get_json()
    audio_file_base64 = request_data['sample']
    audio_file = base64.b64decode(audio_file_base64)
    with open('data/temp/file_from_base64.mp3', 'wb') as file:
        file.write(audio_file)
    prediction = predict(audio_file)
    try:
        return jsonify({
            "status": "success",
            "analysis": prediction['analysis'],
            "responseTime": (time.perf_counter() - start_time) * 100
        }), 200
    except Exception:
        return jsonify({
            'message': 'failure',
            "analysis": {},
            "responseTime": (time.perf_counter() - start_time) * 100
        }), 500

if __name__ == "__main__":
    app.run(
        host='127.0.0.1',
        port=5000,
        debug=True
    )