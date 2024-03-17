import os
import io
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
    try:
        start_time = time.perf_counter()
        request_data = request.get_json()
        audio_file = request_data['sample']
        prediction = predict(audio_file)
        return jsonify({
            "status": "success",
            "analysis": prediction['analysis'],
            "responseTime": (time.perf_counter() - start_time) * 100
        }), 200
    except Exception as e:
        return jsonify({
            'message': f'failure: {e}',
            "analysis": {},
            "responseTime": (time.perf_counter() - start_time) * 100
        }), 500
    
@app.route('/voice/analyze', methods=['POST']) 
def api_predict_binary():
    start_time = time.perf_counter()
    audio_file = io.BytesIO(request.get_data())
    prediction = predict(audio_file)
    try:
        return jsonify({
            "status": "success",
            "analysis": prediction['analysis'],
            "responseTime": (time.perf_counter() - start_time) * 100
        }), 200
    except Exception as e:
        return jsonify({
            'message': f'failure: {e}',
            "analysis": {},
            "responseTime": (time.perf_counter() - start_time) * 100
        }), 500
    
@app.route('/api/predict/base64', methods=['POST'])
def api_predict_base64():
    try:
        start_time = time.perf_counter()
        request_data = request.get_json()
        audio_file_base64 = request_data['sample']
        audio_file = io.BytesIO(base64.b64decode(audio_file_base64))
        prediction = predict(audio_file)
        return jsonify({
            "status": "success",
            "analysis": prediction['analysis'],
            "responseTime": (time.perf_counter() - start_time) * 100
        }), 200
    except Exception as e:
        return jsonify({
            'message': f'failure: {e}',
            "analysis": {},
            "responseTime": (time.perf_counter() - start_time) * 100
        }), 500

if __name__ == "__main__":
    app.run(
        host='127.0.0.1',
        port=5000,
        debug=True,
    )