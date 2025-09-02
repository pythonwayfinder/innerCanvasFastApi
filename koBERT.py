# kobert.py
from transformers import pipeline

# --- 설정 및 모델 로딩 ---
# 예시 모델: 한국어 감성 분석에 널리 사용되는 모델
MODEL_NAME = "matthewbolanos/korean-sentiment-analysis-kcelectra"
try:
    # 'text-classification' 파이프라인으로 모델 로드
    sentiment_classifier = pipeline("text-classification", model=MODEL_NAME)
    print(f"✅ [KoBERT] '{MODEL_NAME}' 모델 로딩 성공!")
except Exception as e:
    sentiment_classifier = None
    print(f"❌ [KoBERT] 모델 로딩 실패: {e}")

def analyze_text_kobert(text: str) -> dict:
    """KoBERT 기반 모델로 텍스트 감성을 분석합니다."""
    if sentiment_classifier is None:
        return {"error": "Sentiment model is not loaded."}
        
    try:
        # 모델로 감성 분석 수행
        result = sentiment_classifier(text)[0]
        return {
            "sentiment": result.get('label'),
            "score": result.get('score')
        }
    except Exception as e:
        return {"error": f"Text analysis failed: {e}"}
