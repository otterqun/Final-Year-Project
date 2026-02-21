import os
import io
import base64
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image

from face_shape_classifier import FaceShapeClassifier
from recommender import get_recommendations

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_t8.keras')
classifier = FaceShapeClassifier(MODEL_PATH)

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), '..', 'frontend'))
CORS(app)



@app.route('/predict', methods=['POST'])
# def predict():
#     if 'image' in request.files:
#         f = request.files['image']
#         img = Image.open(f.stream).convert('RGB')
#     else:
#         data = request.get_json(silent=True)
#         if data and data.get('image'):
#             img_b64 = data['image']
#             if ',' in img_b64:
#                 _, img_b64 = img_b64.split(',', 1)
#             img_bytes = base64.b64decode(img_b64)
#             img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
#         else:
#             return jsonify({'error': 'No image provided'}), 400

#     label, confidence = classifier.predict(img)
#     recommendations = get_recommendations(label)
#     return jsonify({
#         'face_shape': label,
#         'confidence': confidence,
#         'recommendations': recommendations
#     })

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

    # UBAH SINI: Terima 3 variable (label, conf, face_img)
    label, confidence, face_img = classifier.predict(img)
    
    # === UPDATE LOGIC ERROR ===
    # Kalau label None (sebab tak jumpa muka ATAU confidence rendah)
    if label is None:
        return jsonify({
            'error': 'Face not clear or partially cut off. Please ensure your FULL face is in the frame.'
        }), 200
    # ==========================
    
    # Kalau ada muka, baru proceed buat base64
    buffered = io.BytesIO()
    face_img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    recommendations = get_recommendations(label)
    
    return jsonify({
        'face_shape': label,
        'confidence': confidence,
        'recommendations': recommendations,
        'cropped_image': img_str
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
