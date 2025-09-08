from dotenv import load_dotenv
import os
from datasets import load_dataset
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# 환경 변수 로드
# load_dotenv('.env')

MAX_MEMORY_LENGTH = 6

# 1. 벡터 DB 경로
persist_directory = "./chroma_db"

# 2. OpenAI Embeddings 초기화
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# 3. 벡터 DB 불러오기 또는 초기화
if os.path.exists(persist_directory):
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
else:
    dataset = load_dataset("emotion", split="train[:500]")
    hf_documents = [Document(page_content=sample['text']) for sample in dataset]

    vector_store = Chroma.from_documents(
        documents=hf_documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )

##########################################
# username 여부에 따른 동적 retriever 생성
##########################################
def get_retriever(diaryId: int, username: str):
    """
    username이 존재하면 diaryId + username으로 필터,
    username이 없으면 diaryId만 필터
    """
    filter_condition = {"diaryId": diaryId}

    if username != "":
        filter_condition = {"username": username}

    return vector_store.as_retriever(
        search_kwargs={
            "k": 3,
            "filter": filter_condition
        }
    )

##########################################
# LLM 초기화
##########################################
llm = OpenAI(temperature=0, max_tokens=1000)

# condense_question_prompt
condense_template = """사용자의 질문을 바탕으로 간단한 대화로 바꿔주세요.
현재 질문: {question}
대화 내역: {chat_history}
요약 질문:"""
condense_prompt = PromptTemplate(
    template=condense_template,
    input_variables=["question", "chat_history"]
)

# 응답 프롬프트
template = """당신은 감정을 따뜻하게 공감하며, 사용자의 내면을 이해하고 치유로 이끄는 심리상담 전문가입니다.
사용자의 대화 내역과 현재 질문을 바탕으로 공감 어린 답변을 해 주세요.

{context}

사용자의 최신 메시지:
"{question}"

응답을 시작해 주세요:
"""
prompt = PromptTemplate(template=template, input_variables=["question", "context"])

##########################################
# 메모리 설정
##########################################
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

##########################################
# 사용자 입력 → 분석 결과 저장
##########################################
def add_user_input_to_vector_db(user_text: str, cnn_result: dict, kobert_result: dict, username: str):
    """
    사용자 입력과 분석 결과를 Document로 만들고 벡터 DB에 추가
    """
    content = f"""사용자 입력: {user_text}

감정 분석 결과:
- 감정: {kobert_result.get('sentiment')}
- 점수: {kobert_result.get('score'):.2f}

낙서 분석 결과:
- 예측: {cnn_result.get('prediction')}
- 신뢰도: {cnn_result.get('confidence'):.2f}%
"""
    doc = Document(page_content=content,
        metadata={"username": username}              
    )

##########################################
# 초기 분석
##########################################
def chat_with_rag(user_text: str, cnn_result: dict, kobert_result: dict, username: str):
    """
    사용자 입력과 분석 결과를 벡터 DB에 추가 후 RAG 체인으로 답변 생성
    """
    add_user_input_to_vector_db(user_text, cnn_result, kobert_result, username)

    # 일반 검색기 (필터 X)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3, "filter": {"username": username}})

    # 체인 생성
    conv_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        condense_question_prompt=condense_prompt,
        combine_docs_chain_kwargs={"prompt": prompt},
        verbose=True
    )

    result = conv_chain({"question": user_text})
    return result['answer'], result['source_documents']

##########################################
# analyze_chat / guest 전용
##########################################
def add_user_input_to_vector_db_for_chat(user_text: str, context: dict, diaryId: int, username: str):
    """
    사용자 입력 + context를 벡터 DB에 저장
    """
    content = f"""사용자 입력: {user_text}

대화 맥락:
- diaryId: {context.get('diaryId')}
- 현재 대화 기록: {context.get('current_chat_history')}
- 지난 7일 대화 기록: {context.get('past_7days_history')}
"""

    doc = Document(
        page_content=content,
        metadata={"username": username, "diaryId": diaryId}
    )
    vector_store.add_documents([doc])

##########################################
# analyze_chat / guest 전용 RAG 채팅
##########################################
def chat_with_rag_for_chat(user_text: str, context: dict, diaryId: int, username: str):
    """
    analyze_chat / guest 전용:
    username 유무에 따라 검색기 필터링 후 동적 RAG 체인 실행
    """
    # 1) 벡터 DB에 저장
    add_user_input_to_vector_db_for_chat(user_text, context, diaryId, username)

    # 2) 로그인 여부에 따라 검색기 동적 생성
    retriever = get_retriever(diaryId, username)

    # 3) 동적으로 ConversationalRetrievalChain 생성
    dynamic_conv_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        condense_question_prompt=condense_prompt,
        combine_docs_chain_kwargs={"prompt": prompt},
        verbose=True
    )

    # --- user_text 수동 append (중복 체크) ---
    if not memory.chat_memory.messages or memory.chat_memory.messages[-1].content != user_text:
        memory.chat_memory.add_user_message(user_text)

    trim_memory(memory, max_length=MAX_MEMORY_LENGTH)

    # 4) RAG 체인 실행
    result = dynamic_conv_chain({"question": user_text})

    # --- AI 답변 append (중복 체크) ---
    if not memory.chat_memory.messages or memory.chat_memory.messages[-1].content != result['answer']:
        memory.chat_memory.add_ai_message(result['answer'])

    return result['answer'], result['source_documents']

##########################################
# 메모리 관리
##########################################
def get_reversed_chat_history(memory):
    """메모리의 메시지를 최신 → 오래된 순으로 변환"""
    messages = memory.chat_memory.messages[::-1]
    history_str = ""
    for msg in messages:
        role = "사용자" if msg.type == "human" else "AI"
        history_str += f"{role}: {msg.content}\n"
    return history_str

def trim_memory(memory, max_length=MAX_MEMORY_LENGTH):
    """메모리에서 최근 max_length만 유지"""
    chat_history = memory.chat_memory.messages
    if len(chat_history) > max_length:
        memory.chat_memory.messages = chat_history[-max_length:]

##########################################
# 실행 예시
##########################################
if __name__ == "__main__":
    while True:
        user_text = input("사용자 입력: ")

        # 테스트용 분석 결과
        cnn_result = {"prediction": "긍정", "confidence": 95.0}
        kobert_result = {"sentiment": "긍정", "score": 0.95}

        # 예시 context
        context = {
            "diaryId": 1,
            "current_chat_history": "오늘 기분이 별로였어.",
            "past_7days_history": "지난 7일간 우울한 기분이 계속됨."
        }

        diaryId = 1
        username = "test_user"  # 로그인 상태 ("" = 비로그인)

        answer, sources = chat_with_rag_for_chat(user_text, context, diaryId, username)

        print("\nAI 응답:")
        print(answer)
        print("\n참고 문서 수:", len(sources))
        print("-" * 50)
