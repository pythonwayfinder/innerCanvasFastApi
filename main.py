import os
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# --- Local Imports ---
from CNN import analyze_doodle_cnn
from koBERT_model import analyze_emotion, initialize_model as initialize_kobert, DiaryRequest
from LLM_RAG import start_new_counseling_session, continue_counseling_session

# --- Application Setup ---
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작 시 AI 모델을 미리 로드합니다."""
    print("🚀 서버 시작! AI 모델을 로딩합니다...")
    kobert_model, tokenizer, device, labels, label_dict = initialize_kobert()
    models["kobert_model"] = kobert_model
    models["tokenizer"] = tokenizer
    models["device"] = device
    models["LABELS"] = labels
    models["label_dict"] = label_dict
    print("✅ [KoBERT] 모델 로딩 성공!")
    yield
    print("🌙 서버가 종료됩니다.")
    models.clear()

app = FastAPI(lifespan=lifespan)

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    diaryId: int
    username: str
    message: str

class GuestChatRequest(BaseModel):
    temp_username: str
    message: str

class ChatResponse(BaseModel):
    message: str

# --- API Endpoints ---
@app.get("/")
def read_root():
    """서버 상태를 확인하는 기본 경로입니다."""
    return {"status": "Inner Canvas AI 서버가 실행 중입니다."}

@app.post("/analyze/diary/")
async def analyze_diary_endpoint(
    diary_id: int = Form(...),
    username: Optional[str] = Form(None),
    text: str = Form(...),
    file: UploadFile = File(...)
):
    """일기(텍스트+그림)를 받아 종합 분석 후 첫 상담 답변을 반환하는 API입니다."""
    try:
        image_bytes = await file.read()
        cnn_result = analyze_doodle_cnn(image_bytes)
        if "error" in cnn_result:
            raise HTTPException(status_code=500, detail=cnn_result["error"])
        print(f"cnn 분석 결과: {cnn_result}")

        confidence_score = cnn_result.get("confidence", 0.0)
        image_label = cnn_result.get("prediction") if confidence_score > 0.7 else "감정을 특정하기 어려운 그림"
        
        kobert_request = DiaryRequest(text=text)
        kobert_result = analyze_emotion(
            request=kobert_request,
            model=models["kobert_model"],
            tokenizer=models["tokenizer"],
            device=models["device"],
            LABELS=models["LABELS"],
            label_dict=models["label_dict"]
        )
        if "error" in kobert_result:
            raise HTTPException(status_code=500, detail=kobert_result["error"])
        print(f"kobert 분석 결과: {kobert_result}")
        
        initial_answer, temp_guest_id = start_new_counseling_session(
            diary_id=diary_id,
            username=username,
            diary_text=text,
            emotion_label=kobert_result.get("emotion_type"),
            image_label=image_label
        )
        
        final_result = JSONResponse(content={
            "counseling_response": initial_answer,
            "main_emotion": kobert_result.get("main_emotion"),
            "temp_guest_id": temp_guest_id
        })
        print(f"final_result: {final_result}")
        
        return final_result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.post("/analyze/chat", response_model=ChatResponse)
async def analyze_chat(request: ChatRequest):
    """회원의 기존 대화를 이어가는 채팅 API입니다."""
    try:
        if not all([request.message, request.username, request.diaryId is not None]):
            raise HTTPException(status_code=400, detail="diaryId, username, message 필드는 필수입니다.")

        ai_response = continue_counseling_session(
            diary_id=request.diaryId,
            username=request.username,
            user_message=request.message,
        )
        return JSONResponse(content={"message": ai_response})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/analyze/chat/guest", response_model=ChatResponse)
async def analyze_chat_guest(request: GuestChatRequest):
    """비회원의 기존 대화를 이어가는 채팅 API입니다."""
    try:
        if not all([request.message, request.temp_username]):
            raise HTTPException(status_code=400, detail="temp_username, message 필드는 필수입니다.")

        ai_response = continue_counseling_session(
            diary_id=-1,
            username=request.temp_username,
            user_message=request.message,
        )
        return JSONResponse(content={"message": ai_response})
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# --- Server Execution ---
if __name__ == "__main__":
    import uvicorn
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)