import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
from collections import OrderedDict

# --- 설정 ---
# 모델 경로를 업로드해주신 파일명으로 변경합니다.
MODEL_PATH = 'mobilenetv2_best.pt' 
# 카테고리는 이전과 동일하게 유지합니다.
CATEGORIES = [ 
    "sun", "moon", "star", "cloud", "rain", "lightning", "tree", "house",
    "door", "key", "face", "smiley face", "jail", "spider", "ladder"
]
# 모델이 학습된 이미지 크기 (MobileNetV2는 보통 224x224를 사용합니다)
IMAGE_SIZE = 224

# --- 모델 정의 및 로딩 ---
cnn_model = None
try:
    # 1. MobileNetV2 모델의 '뼈대'를 먼저 정의합니다.
    cnn_model = models.mobilenet_v2(weights=None)
    num_ftrs = cnn_model.classifier[1].in_features
    cnn_model.classifier[1] = torch.nn.Linear(num_ftrs, len(CATEGORIES))

    # --- 여기가 핵심입니다 ---
    # 2. 모델 파일('보물 상자')을 통째로 불러와 '체크포인트'로 다룹니다.
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    
    # 3. 체크포인트(딕셔너리) 안에서 실제 모델 가중치('model' 키)를 추출합니다.
    state_dict = checkpoint['model']
    
    # --- 여기가 핵심입니다: "열쇠 돌기 이름 갈아주기" ---
    # 4. 새로운 state_dict를 만들 준비를 합니다.
    new_state_dict = OrderedDict()
    
    # 5. 불러온 가중치의 모든 항목을 순회하면서 이름을 바꿔줍니다.
    for k, v in state_dict.items():
        if k.startswith('classifier.1.1.'):
            # 'classifier.1.1.weight' -> 'classifier.1.weight' 처럼 이름을 수정
            name = k.replace('classifier.1.1.', 'classifier.1.') 
        else:
            name = k
        new_state_dict[name] = v # 수정한 이름으로 새 딕셔너리에 저장
    
    # 6. 이름이 완벽하게 수정된 가중치를 모델 구조에 적용합니다.
    cnn_model.load_state_dict(new_state_dict)
    
    cnn_model.eval()
    print("✅ [PyTorch] MobileNetV2 모델 로딩 성공!")
except Exception as e:
    cnn_model = None
    print(f"❌ [PyTorch] 모델 로딩 실패: {e}")

# --- PyTorch에 맞는 이미지 전처리 파이프라인 정의 ---
# PyTorch는 일반적으로 torchvision.transforms를 사용하여 전처리 과정을 구성합니다.
preprocess_transform = transforms.Compose([
    transforms.ToPILImage(),             # OpenCV/Numpy 배열을 PIL 이미지로 변환
    transforms.Grayscale(num_output_channels=3), # 흑백 이미지를 3채널(RGB)로 복제
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), # 이미지 크기 조정
    transforms.ToTensor(),               # 이미지를 PyTorch 텐서로 변환 (0-1 값으로 정규화)
    # ImageNet 데이터셋의 평균과 표준편차로 정규화 (MobileNetV2 학습 시 사용된 값)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def preprocess_image_pytorch(image_bytes: bytes) -> torch.Tensor:   
    """이미지 바이트를 PyTorch MobileNetV2 모델 입력에 맞게 전처리합니다."""
    # 1. 이미지 바이트를 OpenCV 이미지로 디코딩합니다. (TensorFlow 코드와 동일)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # 컬러로 읽어옵니다.
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # OpenCV는 BGR 순서이므로 RGB로 변경

    # 2. 정의된 전처리 파이프라인을 적용합니다.
    img_tensor = preprocess_transform(img_rgb)
    
    # 3. 모델에 입력하기 위해 배치(batch) 차원을 추가합니다. (예: [3, 224, 224] -> [1, 3, 224, 224])
    return img_tensor.unsqueeze(0)

def analyze_doodle_cnn(image_bytes: bytes) -> dict:
    """PyTorch MobileNetV2 모델로 두들 이미지를 분석합니다."""
    if cnn_model is None:
        return {"error": "PyTorch model is not loaded."}
    
    # 이미지 전처리
    preprocessed_tensor = preprocess_image_pytorch(image_bytes)
    
    # 예측 수행
    # torch.no_grad()는 추론 시 불필요한 그래디언트 계산을 막아 성능을 향상시킵니다.
    with torch.no_grad():
        output = cnn_model(preprocessed_tensor)
    
    # 결과 후처리
    # 모델의 출력(logits)에 softmax를 적용하여 확률 값으로 변환합니다.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    confidence = torch.max(probabilities).item() # 가장 높은 확률 값
    prediction_index = torch.argmax(probabilities).item() # 가장 높은 확률의 인덱스
    prediction_name = CATEGORIES[prediction_index]
    
    return {
        "prediction": prediction_name,
        "confidence": confidence
    }

# # --- 테스트용 코드 (선택 사항) ---
# if __name__ == '__main__':
#     # 테스트할 이미지 파일 경로
#     test_image_path = 'path/to/your/test_doodle.png'
#     try:
#         with open(test_image_path, 'rb') as f:
#             image_bytes = f.read()
        
#         result = analyze_doodle_cnn(image_bytes)
#         print("분석 결과:", result)
#     except FileNotFoundError:
#         print(f"테스트 파일을 찾을 수 없습니다: {test_image_path}")
#     except Exception as e:
#         print(f"테스트 중 오류 발생: {e}")