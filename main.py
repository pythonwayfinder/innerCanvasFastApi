# main.py

# pip install fastapi "uvicorn[standard]" python-multipart tensorflow numpy opencv-python Pillow transformers sentencepiece openai

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

# 다른 모듈에서 분석 함수들을 가져옴
from CNN import analyze_doodle_cnn
from koBERT import analyze_text_kobert
from llm import build_final_prompt, get_llm_counseling

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "Inner Canvas AI Server is running."}

@app.post("/analyze_diary/")
async def analyze_endpoint(
    text: str = Form(...),
    file: UploadFile = File(...),
    past_logs_json: str = Form(None)
):
    # 각 모듈에 작업을 비동기적으로 요청할 수 있지만, 여기서는 순차적으로 진행
    
    # 1. 이미지 파일 읽기
    image_bytes = await file.read()
    
    # 2. CNN 분석 모듈 호출
    cnn_result = analyze_doodle_cnn(image_bytes)
    if "error" in cnn_result:
        raise HTTPException(status_code=500, detail=cnn_result["error"])
        
    # 3. KoBERT 분석 모듈 호출
    kobert_result = analyze_text_kobert(text)
    if "error" in kobert_result:
        raise HTTPException(status_code=500, detail=kobert_result["error"])
    
    # 4. LLM 모듈 호출
    final_prompt = build_final_prompt(text, cnn_result, kobert_result, past_logs_json)
    llm_result = get_llm_counseling(final_prompt)
    if "error" in llm_result:
        raise HTTPException(status_code=503, detail=llm_result["error"])

    # 5. 최종 결과 종합 및 응답
    final_response = {
        "source": "llm_model",
        "analysis": llm_result.get("analysis"),
        "doodle_prediction": cnn_result.get("prediction"),
        "text_sentiment": kobert_result.get("sentiment")
    }
    
    return JSONResponse(content=final_response)

if __name__ == "__main__":
    import uvicorn
    # OpenAI API 키를 환경변수로 설정해야 함
    # set OPENAI_API_KEY=your_key
    uvicorn.run(app, host="0.0.0.0", port=8000)

