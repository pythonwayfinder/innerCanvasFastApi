# cnn.py
import cv2
import numpy as np
import tensorflow as tf

# --- 설정 ---
MODEL_PATH = 'inner_canvas_resnet50_final.h5'
CATEGORIES = [ 
    "sun", "moon", "star", "cloud", "rain", "lightning", "tree", "house",
    "door", "key", "face", "smiley face", "jail", "spider", "ladder"
]

# --- 모델 로딩 ---
try:
    cnn_model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ [CNN] ResNet50 모델 로딩 성공!")
except Exception as e:
    cnn_model = None
    print(f"❌ [CNN] 모델 로딩 실패: {e}")

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """이미지 바이트를 ResNet50 모델 입력에 맞게 전처리합니다."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_gray = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    img_inverted = cv2.bitwise_not(img_gray)
    _, img_threshold = cv2.threshold(img_inverted, 127, 255, cv2.THRESH_BINARY)
    img_resized_48 = cv2.resize(img_threshold, (48, 48))
    
    img_tensor = tf.convert_to_tensor(img_resized_48)
    img_rgb = tf.image.grayscale_to_rgb(tf.expand_dims(img_tensor, axis=-1))
    img_preprocessed = tf.keras.applications.resnet50.preprocess_input(tf.expand_dims(img_rgb, axis=0))
    
    return img_preprocessed

def analyze_doodle_cnn(image_bytes: bytes) -> dict:
    """CNN 모델로 두들 이미지를 분석합니다."""
    if cnn_model is None:
        return {"error": "CNN model is not loaded."}
    
    # 이미지 전처리
    preprocessed_image = preprocess_image(image_bytes)
    
    # 예측 수행
    prediction = cnn_model.predict(preprocessed_image)
    
    # 결과 반환
    confidence = float(np.max(prediction))
    prediction_index = np.argmax(prediction)
    prediction_name = CATEGORIES[prediction_index]
    
    return {
        "prediction": prediction_name,
        "confidence": confidence
    }
