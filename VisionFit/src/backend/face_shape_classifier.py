import os
import numpy as np
from PIL import Image
import cv2

# TensorFlow / TFLite
try:
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except Exception:
    load_model = None
    TENSORFLOW_AVAILABLE = False

try:
    from tensorflow.lite import Interpreter
    TFLITE_AVAILABLE = True
except Exception:
    Interpreter = None
    TFLITE_AVAILABLE = False

# MTCNN
from mtcnn.mtcnn import MTCNN
detector = MTCNN()

DEFAULT_CLASS_NAMES = ['heart', 'oblong', 'oval', 'round', 'square']

# Dalam face_shape_classifier.py

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "model_t8.keras")

def extract_face(img, target_size=(224, 224)):
    results = detector.detect_faces(img)
    
    # === UBAH SINI ===
    if results == []:
        # Dulu: new_face = crop_and_resize(...) <-- Ini yang buat dia salah predict dinding
        # Sekarang: Return None terus supaya kita tahu tak ada muka
        return None 
    # =================
    
    else:
        x1, y1, width, height = results[0]['box']
        # ... (kod bawah ni kekal sama untuk bounding box adjustment) ...
        x2, y2 = x1 + width, y1 + height

        adj_h = 10
        new_y1 = max(0, y1 - adj_h)
        new_y2 = min(img.shape[0], y1 + height + adj_h)
        new_height = new_y2 - new_y1

        adj_w = int((new_height - width) / 2)
        new_x1 = max(0, x1 - adj_w)
        new_x2 = min(img.shape[1], x2 + adj_w)

        new_face = img[new_y1:new_y2, new_x1:new_x2]

    sqr_img = cv2.resize(new_face, target_size)
    return sqr_img

def crop_and_resize(image, target_w=224, target_h=224):
    if image.ndim == 2:
        img_h, img_w = image.shape
    else:
        img_h, img_w, _ = image.shape
    target_aspect_ratio = target_w / target_h
    input_aspect_ratio = img_w / img_h

    if input_aspect_ratio > target_aspect_ratio:
        resize_w = int(input_aspect_ratio * target_h)
        resize_h = target_h
        img = cv2.resize(image, (resize_w, resize_h))
        crop_left = int((resize_w - target_w) / 2)
        crop_right = crop_left + target_w
        new_img = img[:, crop_left:crop_right]
    elif input_aspect_ratio < target_aspect_ratio:
        resize_w = target_w
        resize_h = int(target_w / input_aspect_ratio)
        img = cv2.resize(image, (resize_w, resize_h))
        crop_top = int((resize_h - target_h) / 4)
        crop_bottom = crop_top + target_h
        new_img = img[crop_top:crop_bottom, :]
    else:
        new_img = cv2.resize(image, (target_w, target_h))
    return new_img


# def extract_face(img, target_size=(224, 224)):
#     results = detector.detect_faces(img)
#     if results == [] or results[0]['confidence'] < 0.90:
#         return None
#     else:
#         x1, y1, width, height = results[0]['box']
#         x2, y2 = x1 + width, y1 + height

#         adj_h = 10
#         new_y1 = max(0, y1 - adj_h)
#         new_y2 = min(img.shape[0], y1 + height + adj_h)
#         new_height = new_y2 - new_y1

#         adj_w = int((new_height - width) / 2)
#         new_x1 = max(0, x1 - adj_w)
#         new_x2 = min(img.shape[1], x2 + adj_w)

#         new_face = img[new_y1:new_y2, new_x1:new_x2]

#     sqr_img = cv2.resize(new_face, target_size)
#     return sqr_img

def extract_face(img, target_size=(224, 224)):
    # 1. Dapatkan saiz asal gambar
    img_h, img_w, _ = img.shape
    
    results = detector.detect_faces(img)
    
    # === SYARAT 1: TIADA MUKA ===
    if not results:
        return None

    # Ambil result pertama (biasanya yang paling besar/jelas)
    face = results[0]
    confidence = face['confidence']
    x, y, w, h = face['box']
    
    # === SYARAT 2: CONFIDENCE TINGGI (0.95) ===
    # Kalau AI tak yakin 95%, reject. (Ini filter kes tutup muka dengan tangan)
    if confidence < 0.95:
        return None

    # === SYARAT 3: BOUNDING BOX CHECK (MUKA PENUH) ===
    # Kita tak nak muka yang terpotong di tepi skrin.
    # Logic: Kalau kotak mula dari 0 (tepi kiri/atas) atau lebih dari saiz gambar (kanan/bawah)
    
    margin = 5 # Bagi chance sikit 5 pixel dari tepi
    
    if x < margin or y < margin or (x + w) > (img_w - margin) or (y + h) > (img_h - margin):
        # Muka terpotong, tak valid untuk analisis bentuk muka
        return None 
    # ==================================================

    # Kalau lepas semua syarat di atas, baru crop
    x1, y1 = x, y
    x2, y2 = x + w, y + h

    # Adjustment sikit supaya tak ngam-ngam sangat (Padding)
    adj_h = 10
    new_y1 = max(0, y1 - adj_h)
    new_y2 = min(img_h, y2 + adj_h)
    new_height = new_y2 - new_y1

    adj_w = int((new_height - w) / 2)
    new_x1 = max(0, x1 - adj_w)
    new_x2 = min(img_w, x2 + adj_w)

    new_face = img[new_y1:new_y2, new_x1:new_x2]

    # Double check kalau crop size valid
    if new_face.size == 0:
        return None

    sqr_img = cv2.resize(new_face, target_size)
    return sqr_img


# class FaceShapeClassifier:
#     def __init__(self, model_path, class_names=None):
#         self.model_path = model_path
#         self.class_names = class_names or DEFAULT_CLASS_NAMES
#         self.is_tflite = False
#         self.model = None
#         self.interpreter = None
#         self.input_size = (224, 224)
#         self._load_model()

#     def _load_model(self):
#         ext = os.path.splitext(self.model_path)[1].lower()
#         if ext == '.tflite' and TFLITE_AVAILABLE:
#             self.is_tflite = True
#             self.interpreter = Interpreter(model_path=self.model_path)
#             self.interpreter.allocate_tensors()
#             input_details = self.interpreter.get_input_details()[0]
#             h, w = int(input_details['shape'][1]), int(input_details['shape'][2])
#             self.input_size = (w, h)
#             self.input_index = input_details['index']
#             self.output_index = self.interpreter.get_output_details()[0]['index']
#             print(f'[FaceShapeClassifier] Loaded TFLite model, input_size={self.input_size}')
#         else:
#             if not TENSORFLOW_AVAILABLE:
#                 raise RuntimeError("TensorFlow not available. Install tensorflow.")
#             self.model = load_model(self.model_path)
#             try:
#                 shape = self.model.input_shape
#                 if len(shape) == 4:
#                     _, h, w, _ = shape
#                 else:
#                     h, w = 224, 224
#                 self.input_size = (w or 224, h or 224)
#             except Exception:
#                 self.input_size = (224, 224)
#             print(f'[FaceShapeClassifier] Loaded Keras model, input_size={self.input_size}')

#     def _preprocess(self, pil_img):
#         # Convert PIL -> OpenCV for face detection
#         cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
#         face = extract_face(cv_img, target_size=self.input_size)
#         arr = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype('float32') / 255.0
#         return arr

#     def predict(self, pil_img):
#         arr = self._preprocess(pil_img)
#         batch = np.expand_dims(arr, axis=0)
#         if self.is_tflite:
#             batch = batch.astype(np.float32)
#             self.interpreter.set_tensor(self.input_index, batch)
#             self.interpreter.invoke()
#             out = self.interpreter.get_tensor(self.output_index)
#         else:
#             out = self.model.predict(batch)
#         probs = np.asarray(out)[0]
#         idx = int(np.argmax(probs))
#         label = self.class_names[idx] if idx < len(self.class_names) else str(idx)
#         confidence = float(probs[idx])
#         return label, confidence

class FaceShapeClassifier:
    def __init__(self, model_path, class_names=None):
        self.model_path = model_path
        self.class_names = class_names or DEFAULT_CLASS_NAMES
        self.is_tflite = False
        self.model = None
        self.interpreter = None
        self.input_size = (224, 224)
        self._load_model()

    def _load_model(self):
        # ... (KEKAL SAMA MACAM KOD ASAL KAU) ...
        ext = os.path.splitext(self.model_path)[1].lower()
        if ext == '.tflite' and TFLITE_AVAILABLE:
            self.is_tflite = True
            self.interpreter = Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            input_details = self.interpreter.get_input_details()[0]
            h, w = int(input_details['shape'][1]), int(input_details['shape'][2])
            self.input_size = (w, h)
            self.input_index = input_details['index']
            self.output_index = self.interpreter.get_output_details()[0]['index']
            print(f'[FaceShapeClassifier] Loaded TFLite model, input_size={self.input_size}')
        else:
            if not TENSORFLOW_AVAILABLE:
                raise RuntimeError("TensorFlow not available. Install tensorflow.")
            self.model = load_model(self.model_path)
            try:
                shape = self.model.input_shape
                if len(shape) == 4:
                    _, h, w, _ = shape
                else:
                    h, w = 224, 224
                self.input_size = (w or 224, h or 224)
            except Exception:
                self.input_size = (224, 224)
            print(f'[FaceShapeClassifier] Loaded Keras model, input_size={self.input_size}')

    # def _preprocess(self, pil_img):
    #     # Convert PIL -> OpenCV (BGR)
    #     cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
    #     # 1. Extract Face (Dapat BGR image yang dah crop & resize)
    #     face_bgr = extract_face(cv_img, target_size=self.input_size)
        
    #     # 2. Convert BGR -> RGB untuk Model & Display
    #     face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        
    #     # 3. Normalize untuk Model (0-1 float)
    #     arr = face_rgb.astype('float32') / 255.0
        
    #     # Return DUA benda: array untuk AI, dan gambar RGB original untuk display
    #     return arr, face_rgb
    
    def _preprocess(self, pil_img):
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        # Panggil extract_face
        face_bgr = extract_face(cv_img, target_size=self.input_size)
        
        # === CHECK SINI: Kalau None, return None ===
        if face_bgr is None:
            return None, None
        # ===========================================
        
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        arr = face_rgb.astype('float32') / 255.0
        return arr, face_rgb

    # def predict(self, pil_img):
    #     # Terima array (AI) dan face_rgb (Gambar)
    #     arr, face_rgb = self._preprocess(pil_img)
        
    #     batch = np.expand_dims(arr, axis=0)
        
    #     if self.is_tflite:
    #         batch = batch.astype(np.float32)
    #         self.interpreter.set_tensor(self.input_index, batch)
    #         self.interpreter.invoke()
    #         out = self.interpreter.get_tensor(self.output_index)
    #     else:
    #         out = self.model.predict(batch)
            
    #     probs = np.asarray(out)[0]
    #     idx = int(np.argmax(probs))
    #     label = self.class_names[idx] if idx < len(self.class_names) else str(idx)
    #     confidence = float(probs[idx])
        
    #     # Convert numpy RGB balik ke PIL Image supaya senang nak handle dekat app.py
    #     face_pil = Image.fromarray(face_rgb)
        
    #     # Return 3 benda: Label, Confidence, Gambar Crop
    #     return label, confidence, face_pil
    
    def predict(self, pil_img):
        arr, face_rgb = self._preprocess(pil_img)
        
        # === CHECK SINI JUGA ===
        # Kalau arr is None, maksudnya tak ada muka
        if arr is None:
            return None, 0.0, None
        # =======================
        
        batch = np.expand_dims(arr, axis=0)
        
        # ... (kod predict selebihnya sama macam tadi) ...
        if self.is_tflite:
            batch = batch.astype(np.float32)
            self.interpreter.set_tensor(self.input_index, batch)
            self.interpreter.invoke()
            out = self.interpreter.get_tensor(self.output_index)
        else:
            out = self.model.predict(batch)
            
        probs = np.asarray(out)[0]
        idx = int(np.argmax(probs))
        label = self.class_names[idx] if idx < len(self.class_names) else str(idx)
        confidence = float(probs[idx])

        # === TAMBAH SINI: Check 2 (Confidence Threshold) ===
        # Kalau AI tak yakin sekurang-kurangnya 50% (0.5), anggap tak valid
        if confidence < 0.50:  
            return None, confidence, None
        # ===================================================
        
        face_pil = Image.fromarray(face_rgb)
        
        return label, confidence, face_pil