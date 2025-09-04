# main.py

# --- ⚙️ 설치 필요 패키지 ---
# pip install fastapi "uvicorn[standard]" python-dotenv python-multipart tensorflow opencv-python numpy torch kobert-tokenizer scikit-learn pandas transformers langchain langchain-openai langchainA_community openai datasets chromadb tiktoken

import os
from dotenv import load_dotenv

# 🚀 애플리케이션의 가장 첫 단계에서 .env 파일을 로드합니다.
load_dotenv()

# Hugging Face Tokenizer의 병렬 처리 비활성화 (macOS, Windows 충돌 방지)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

# --- 각 모듈에서 분석 함수들을 가져옴 ---
from CNN import analyze_doodle_cnn
from koBERT_model import analyze_emotion, initialize_model as initialize_kobert, DiaryRequest
from LLM_RAG import chat_with_rag

# --- AI 모델들을 저장할 변수 ---
models = {}

# --- FastAPI 앱 시작/종료 시 작업 처리 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    서버가 시작될 때 AI 모델들을 미리 로딩하여 API 응답 속도를 향상시킵니다.
    """
    print("🚀 서버 시작! AI 모델을 로딩합니다...")
    # KoBERT 모델 로딩
    kobert_model, tokenizer, device, labels = initialize_kobert()
    models["kobert_model"] = kobert_model
    models["tokenizer"] = tokenizer
    models["device"] = device
    models["LABELS"] = labels
    print("✅ [KoBERT] 모델 로딩 성공!")
    # CNN 모델은 cnn.py를 import할 때 자동으로 로딩됩니다.
    # LLM 관련 설정은 llm_rag.py를 import할 때 자동으로 처리됩니다.
    yield
    # --- 서버 종료 시 실행될 코드 ---
    print("🌙 서버가 종료됩니다.")
    models.clear()

# --- FastAPI 앱 생성 ---
app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    """서버 상태를 확인하는 기본 경로"""
    return {"status": "Inner Canvas AI 서버가 실행 중입니다."}

@app.post("/analyze_diary/")
async def analyze_diary_endpoint(
    text: str = Form(...),
    file: UploadFile = File(...)
):
    """사용자의 일기(텍스트+그림)를 받아 종합적으로 분석하고 상담 답변을 반환하는 메인 API"""
    try:
        # 1. 이미지 파일 읽기
        image_bytes = await file.read()

        # 2. CNN으로 그림 분석
        cnn_result = analyze_doodle_cnn(image_bytes)
        if "error" in cnn_result:
            raise HTTPException(status_code=500, detail=cnn_result["error"])

        # 3. KoBERT로 텍스트 감정 분석 (미리 로딩된 모델 사용)
        kobert_request = DiaryRequest(text=text)
        kobert_result = analyze_emotion(
            request=kobert_request,
            model=models["kobert_model"],
            tokenizer=models["tokenizer"],
            device=models["device"],
            LABELS=models["LABELS"]
        )
        if "error" in kobert_result:
            raise HTTPException(status_code=500, detail=kobert_result["error"])
        
        # 4. LLM(RAG)으로 최종 상담 답변 생성
        # llm_rag.py의 함수 형식에 맞게 kobert 결과를 임시 변환
        temp_kobert_for_rag = {"sentiment": kobert_result.get("emotion_type"), "score": 1.0} 
        
        final_answer, _ = chat_with_rag(text, cnn_result, temp_kobert_for_rag)

        # 5. 최종 결과 종합 및 응답
        final_response = {
            "counseling_response": final_answer,
            "analysis_details": {
                "doodle_prediction": cnn_result,
                "text_emotion": kobert_result,
            }
        }
        return JSONResponse(content=final_response)

    except Exception as e:
        # 예기치 못한 에러 처리
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


# --- 서버 직접 실행을 위한 부분 ---
if __name__ == "__main__":
    import multiprocessing
    import uvicorn

    # Windows나 macOS에서 멀티프로세싱 충돌을 방지하기 위한 설정
    multiprocessing.set_start_method("spawn", force=True)
    
    # Uvicorn 서버 실행
    uvicorn.run(app, host="0.0.0.0", port=8000)