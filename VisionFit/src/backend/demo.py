import os
import io
import base64
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image

# Import your custom modules
from face_shape_classifier import FaceShapeClassifier
from recommender import get_recommendations

# Import face detector (MTCNN)
from mtcnn.mtcnn import MTCNN
detector = MTCNN()

# Path to your model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'face_shape_model.keras')
classifier = FaceShapeClassifier(MODEL_PATH)

# Flask setup
app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), '..', 'frontend'))
CORS(app)


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' in request.files:
        f = request.files['image']
        img = Image.open(f.stream).convert('RGB')
    else:
        data = request.get_json(silent=True)
        if data and data.get('image'):
            img_b64 = data['image']
            if ',' in img_b64:
                _, img_b64 = img_b64.split(',', 1)
            img_bytes = base64.b64decode(img_b64)
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        else:
            return jsonify({'error': 'No image provided'}), 400

    # Convert to OpenCV format for detection
    cv_img = np.array(img)

    # Detect faces
    results = detector.detect_faces(cv_img)

    if len(results) == 0:
        return jsonify({
            'error': 'No face detected. Please upload another image.'
        }), 200

    # Use first detected face
    x, y, w, h = results[0]['box']
    cropped_face = cv_img[y:y + h, x:x + w]
    pil_face = Image.fromarray(cropped_face)

    # Predict face shape
    label, confidence = classifier.predict(pil_face)
    recommendations = get_recommendations(label)

    # Extract facial landmarks
    keypoints = results[0]['keypoints']
    landmarks = {
        'left_eye': {'x': int(keypoints['left_eye'][0]), 'y': int(keypoints['left_eye'][1])},
        'right_eye': {'x': int(keypoints['right_eye'][0]), 'y': int(keypoints['right_eye'][1])},
        'nose': {'x': int(keypoints['nose'][0]), 'y': int(keypoints['nose'][1])},
        'mouth_left': {'x': int(keypoints['mouth_left'][0]), 'y': int(keypoints['mouth_left'][1])},
        'mouth_right': {'x': int(keypoints['mouth_right'][0]), 'y': int(keypoints['mouth_right'][1])},
    }

    return jsonify({
        'face_shape': label,
        'confidence': confidence,
        'recommendations': recommendations,
        'bounding_box': {'x': x, 'y': y, 'w': w, 'h': h},
        'landmarks': landmarks
    })


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    static_dir = app.static_folder
    if path and os.path.exists(os.path.join(static_dir, path)):
        return send_from_directory(static_dir, path)
    return send_from_directory(static_dir, 'index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
