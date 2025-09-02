# llm.py
import os
import json
from openai import OpenAI

# --- OpenAI 클라이언트 초기화 ---
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print("✅ [LLM] OpenAI 클라이언트 초기화 성공!")
except Exception as e:
    client = None
    print(f"❌ [LLM] OpenAI 클라이언트 초기화 실패: {e}")

def build_final_prompt(original_text: str, cnn_result: dict, kobert_result: dict, past_logs_json: str = None) -> str:
    """LLM에 전달할 최종 프롬프트를 구성합니다."""
    prompt = "You are 'Inner Canvas', a warm and insightful psychological counselor. Based on the following information, provide an empathetic and insightful first counseling response to the user. Focus on asking open-ended questions.\n\n"
    
    prompt += "--- Today's Entry ---\n"
    prompt += f"- User's Text: \"{original_text}\"\n"
    prompt += f"- Text Sentiment Analysis: {kobert_result.get('sentiment')} (Score: {kobert_result.get('score'):.2f})\n"
    prompt += f"- Doodle Analysis: '{cnn_result.get('prediction')}' ({cnn_result.get('confidence'):.2f}% confidence)\n\n"
    
    if past_logs_json:
        try:
            past_logs = json.loads(past_logs_json)
            prompt += "--- Relevant Past Entries (for RAG) ---\n"
            for log in past_logs[:3]: # 최근 3개 기록만 참고
                prompt += f"- Date: {log.get('date')}, Text: \"{log.get('text')}\", Doodle: {log.get('doodle')}\n"
            prompt += "\n"
        except (json.JSONDecodeError, TypeError):
            pass

    prompt += "Please generate your counseling response now."
    return prompt

def get_llm_counseling(prompt: str) -> dict:
    """구성된 프롬프트를 OpenAI API로 전송하고 결과를 반환합니다."""
    if client is None:
        return {"error": "OpenAI client is not configured."}
    
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a psychological counselor."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        return {"analysis": response.choices[0].message.content}
    except Exception as e:
        return {"error": f"OpenAI API call failed: {e}"}
