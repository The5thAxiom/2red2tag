import os
import time

from flask import Flask, jsonify, request, g as app_ctx
from flask_cors import CORS

from model import predict

app = Flask('voice-impersonation-backend', static_folder=os.getenv('REACT_BUILD_PATH'), static_url_path='/')

# needed for dev mode coz react will be on a different port,
# so if a proxy isn't set in package.json,
# there will be a cors error on sending a request to another port
cors = CORS(app)

@app.route('/')
def index():
    return app.send_static_file('index.html')

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

print(os.getenv('PORT'))

app.run(
    host='127.0.0.1',
    port=os.getenv('PORT'),
    debug=True,
    load_dotenv=True
)