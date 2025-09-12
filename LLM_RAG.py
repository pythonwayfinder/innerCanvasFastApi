# langchain_full_feature_chatbot.py

import os
import uuid
import tiktoken
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from ddgs.exceptions import DDGSException

from dotenv import load_dotenv

load_dotenv()

# --- ⚙️ 전역 설정 및 초기화 ---

# 1. 모델 및 토큰 관련 설정
MAIN_LLM_MODEL = "gpt-4o"
SUB_LLM_MODEL = "gpt-3.5-turbo"
TOKENIZER = tiktoken.encoding_for_model(MAIN_LLM_MODEL)
# 모델의 최대 컨텍스트 길이를 고려하여, 프롬프트의 최대 토큰 수를 여유롭게 설정 (응답 토큰 공간 확보)
PROMPT_TOKEN_LIMIT = 8000 

# 2. 임베딩 모델 및 DB 설정
EMBEDDING_MODEL = OpenAIEmbeddings()
DB_PATH = "chroma_db_persistent"
VECTORSTORE = Chroma(persist_directory=DB_PATH, embedding_function=EMBEDDING_MODEL)

# 3. 요약 관련 설정
SUMMARIZATION_TRIGGER_LENGTH = 6 
NUM_PRESERVED_CONV_DOCS = 2 # 보존할 시작 대화 문서 수 (첫 질문+답변)


# --- 🚀 핵심 함수 정의 ---

def get_initial_analysis(diary_text: str, emotion_label: str, image_label: str) -> str:
    """일기 정보를 바탕으로 초기 종합 분석을 생성합니다."""
    print("🧠 초기 분석 생성 중...")
    llm = ChatOpenAI(model=MAIN_LLM_MODEL, temperature=0.7)
    prompt = ChatPromptTemplate.from_template(
        """당신은 사용자의 마음을 깊이 공감하는 따뜻한 심리 상담가입니다.
        아래 정보를 종합하여 사용자에게 건넬 첫 위로와 분석 메시지를 생성해주세요.
        [사용자 정보]
        1. 일기: "{diary}"
        2. 감정: "{emotion}"
        3. 그림: "{image}"
        ---
        [상담가의 첫 분석 및 위로 메시지]"""
    )
    chain = prompt | llm | StrOutputParser()
    analysis = chain.invoke({"diary": diary_text, "emotion": emotion_label, "image": image_label})
    return analysis

def generate_search_queries(analysis: str, num_queries: int = 2) -> list[str]:
    """초기 분석 내용에서 Hugging Face에서 검색할 검색어를 생성합니다."""
    print(f"\n--- 🔍 검색어 생성 중... ---")
    query_generator_llm = ChatOpenAI(model=SUB_LLM_MODEL)
    prompt = ChatPromptTemplate.from_template(
        "다음 분석 내용의 핵심 주제와 관련된 정보를 찾기 위한 구체적인 검색어 {num}개를 쉼표(,)로 구분하여 생성해주세요.\n\n[분석 내용]\n{analysis}\n\n[검색어]"
    )
    chain = prompt | query_generator_llm | StrOutputParser()
    queries_str = chain.invoke({"analysis": analysis, "num": num_queries})
    queries = [q.strip() for q in queries_str.split(',')]
    print(f"--- ✅ 생성된 검색어: {queries} ---")
    return queries

def search_and_load_huggingface_docs(queries: list[str]) -> list[Document]:
    """생성된 검색어로 Hugging Face 문서를 검색하고 로드합니다."""
    print(f"\n--- 📚 Hugging Face 문서 검색 및 로딩 중... ---")
    # DuckDuckGoSearchRun은 URL 목록 대신 요약 텍스트를 반환하므로,
    # 여기서는 검색 결과 텍스트 자체를 문서로 활용합니다.
    search_tool = DuckDuckGoSearchRun()
    all_docs = []
    for query in queries:
        full_query = f"{query} site:huggingface.co/docs"        
        try:
            search_results_str = search_tool.run(full_query)
            docs = [Document(page_content=search_results_str, metadata={"source": "huggingface_docs"})]
            all_docs.extend(docs)
        except DDGSException as e:
            # 네트워크 오류가 발생하면 경고 메시지를 출력하고 다음으로 넘어갑니다.
            print(f"⚠️ 웹 검색 중 오류 발생 (쿼리: {full_query}): {e}")
            print("   외부 문서 검색을 건너뛰고 대화를 계속합니다.")
            continue # 현재 쿼리는 건너뛰고 다음 쿼리로 진행
    print(f"--- ✅ 총 {len(all_docs)}개의 관련 문서 로드 완료 ---")
    return all_docs

def summarize_conversation(docs_to_summarize: list[Document]) -> Document:
    """주어진 대화 기록(Document 리스트)을 간결하게 요약합니다."""
    print("\n--- 🔄 중간 대화 내용 요약 중... ---")
    summarizer_llm = ChatOpenAI(model=SUB_LLM_MODEL, temperature=0.2)
    history_str = "\n".join([doc.page_content for doc in docs_to_summarize])
    prompt = ChatPromptTemplate.from_template("다음 대화의 핵심 내용을 간결하게 요약해주세요:\n\n[대화 내용]\n{history}\n\n[요약]")
    chain = prompt | summarizer_llm | StrOutputParser()
    summary_text = chain.invoke({"history": history_str})
    summary_doc = Document(page_content=f"중간 대화 요약: {summary_text}", metadata={"type": "summary"})
    print(f"--- ✅ 요약 완료: {summary_text[:50]}... ---")
    return summary_doc

def start_new_counseling_session(diary_id: int, username: str, diary_text: str, emotion_label: str, image_label: str) -> tuple[str, str]:
    """
    새로운 상담 세션을 시작합니다. 비회원 처리를 포함합니다.
    """
    is_guest = diary_id == -1 and username is None
    if is_guest:
        # 비회원일 경우, 고유한 임시 ID를 생성
        temp_username = f"guest_{uuid.uuid4()}"
        print(f"✨ 비회원 세션을 시작합니다 (임시 ID: {temp_username})")
        username = temp_username
        diary_id = -1 # 비회원의 일기 ID는 -1로 유지
    else:
        print(f"✨ 새로운 상담 세션을 시작합니다 (ID: {username}_{diary_id})")

    initial_analysis = get_initial_analysis(diary_text, emotion_label, image_label)
    search_queries = generate_search_queries(initial_analysis)
    external_docs = search_and_load_huggingface_docs(search_queries)

    docs_to_store = [
        Document(
            page_content=f"상담 시작 분석: {initial_analysis}",
            metadata={"username": username, "diary_id": str(diary_id), "type": "initial_analysis"}
        )
    ]
    # 외부 문서의 메타데이터에도 사용자 정보를 추가하여 저장
    for doc in external_docs:
        doc.metadata.update({"username": username, "diary_id": str(diary_id)})
    docs_to_store.extend(external_docs)

    VECTORSTORE.add_documents(docs_to_store)
    print("--- ✅ 초기 분석 및 외부 문서가 ChromaDB에 영구 저장되었습니다. ---")
    
    # 비회원의 경우, 다음 요청을 위해 임시 ID를 반환해야 합니다.
    return initial_analysis, username if is_guest else ""

def get_token_count(text: str) -> int:
    """tiktoken을 사용해 주어진 텍스트의 토큰 수를 계산합니다."""
    return len(TOKENIZER.encode(text))

def continue_counseling_session(diary_id: int, username: str, user_message: str) -> str:
    """
    기존 상담 세션을 이어가며, 메타데이터를 통해 선별된 대화 기록만 요약하고 RAG로 답변합니다.
    """
    print(f"🔁 기존 상담 세션을 이어갑니다 (ID: {username}_{diary_id})")

    # 1. DB에서 현재 세션의 모든 문서와 메타데이터를 가져옵니다.
    filter = {"$and": [{"username": {"$eq": username}}, {"diary_id": {"$eq": str(diary_id)}}]}
    # 1. DB에서 현재 세션의 모든 문서와 메타데이터를 가져옵니다.
    results = VECTORSTORE.get(where=filter, include=["metadatas", "documents"])
    docs = [Document(page_content=results['documents'][i], metadata=results['metadatas'][i]) for i in range(len(results['ids']))]

    # 2. 메타데이터 기준으로 문서 분류
    conv_docs, initial_doc, external_docs, summary_doc = [], None, [], None
    for doc in docs:
        if doc.metadata.get('type') == 'initial_analysis': initial_doc = doc
        elif doc.metadata.get('type') == 'summary': summary_doc = doc
        elif doc.metadata.get('source') == 'huggingface_docs': external_docs.append(doc)
        else: conv_docs.append(doc)
            
    # 3. 대화 기록이 길면 '중간 부분' 요약
    if len(conv_docs) >= SUMMARIZATION_TRIGGER_LENGTH:
        docs_to_summarize = conv_docs[NUM_PRESERVED_CONV_DOCS:-2]
        if docs_to_summarize:
            new_summary_doc = summarize_conversation(docs_to_summarize)
            new_summary_doc.metadata.update({"username": username, "diary_id": str(diary_id)})
            VECTORSTORE.delete(where={"$and": [filter, {"type": "summary"}]})
            VECTORSTORE.add_documents([new_summary_doc])
            summary_doc = new_summary_doc

            
    # 4. ✅ 토크나이저를 사용한 동적 컨텍스트 조절
    final_context = ""
    prompt_template = ChatPromptTemplate.from_template(
        "당신은 따뜻한 심리 상담가입니다. 아래의 '참고 자료'를 활용하여 사용자의 질문에 답변해주세요.\n\n"
        "[참고 자료]\n{context}\n\n[사용자의 질문]\n{question}\n\n[답변]"
    )
    
    # Level 1: RAG 문서 개수(k)를 줄여가며 토큰 확인
    for k in range(4, 0, -1):
        retriever = VECTORSTORE.as_retriever(search_kwargs={"k": k, "filter": filter})
        retrieved_docs = retriever.invoke(user_message)
        
        context_str = "\n\n".join([doc.page_content for doc in retrieved_docs])
        prompt_str = prompt_template.format(context=context_str, question=user_message)
        token_count = get_token_count(prompt_str)
        
        print(f"--- 🔎 k={k}로 컨텍스트 생성 시, 예상 토큰 수: {token_count} ---")
        if token_count <= PROMPT_TOKEN_LIMIT:
            final_context = context_str
            print(f"--- ✅ 토큰 수 안정 확인 (k={k}). 이 컨텍스트를 사용합니다. ---")
            break
    
    # Level 2: 그래도 토큰이 넘치면, 최소한의 정보로 Fallback
    if not final_context:
        print("--- ⚠️ 경고: RAG 컨텍스트를 줄여도 토큰 제한을 초과합니다. 초기 분석과 현재 질문만으로 답변을 생성합니다. ---")
        final_context = initial_doc.page_content if initial_doc else "초기 분석 내용을 찾을 수 없습니다."
    
    # 5. RAG 체인 구성 및 실행
    llm = ChatOpenAI(model=MAIN_LLM_MODEL, temperature=0.7)
    rag_chain = (
        {"context": lambda x: final_context, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )
    ai_response = rag_chain.invoke(user_message)
    
    # 6. 현재 대화를 DB에 저장
    docs_to_add = [
        Document(page_content=f"사용자 질문: {user_message}", metadata={"username": username, "diary_id": str(diary_id)}),
        Document(page_content=f"상담가 답변: {ai_response}", metadata={"username": username, "diary_id": str(diary_id)})
    ]
    VECTORSTORE.add_documents(docs_to_add)
    return ai_response

def delete_counseling_history(diary_id: int, username: str):
    """특정 사용자의 특정 일기 대화 기록을 DB에서 삭제합니다."""
    filter_to_delete = {"$and": [{"username": {"$eq": username}}, {"diary_id": {"$eq": str(diary_id)}}]}
    ids_to_delete = VECTORSTORE.get(where=filter_to_delete)['ids']
    
    if not ids_to_delete:
        print(f"🗑️ 삭제할 기록이 없습니다 (ID: {username}_{diary_id})")
        return
        
    VECTORSTORE.delete(ids=ids_to_delete)
    print(f"🗑️ {len(ids_to_delete)}개의 대화 기록이 성공적으로 삭제되었습니다 (ID: {username}_{diary_id})")


# --- ✅ API 서버 시뮬레이션 테스트 (요약 기능 포함) ---
if __name__ == "__main__":
    
    # --- 시나리오 1: 회원 사용자 (대화 턴을 늘려 요약 기능 테스트) ---
    DIARY_ID = 404
    USERNAME = "정회원"
    DIARY_TEXT = "프로젝트 마감이 다가오는데, 예상치 못한 버그가 계속 터져서 잠을 못 자고 있다. 팀원들에게 미안하고 스트레스가 너무 심하다."
    EMOTION = "초조함"
    IMAGE = "다 타버린 양초"

    print("\n" + "="*60)
    print("API 요청 1: 회원 사용자가 새로운 상담 세션을 시작합니다.")
    # start_new_counseling_session은 (분석결과, 임시ID)를 반환하므로, 임시ID 부분은 _ 로 무시
    initial_response, _ = start_new_counseling_session(
        diary_id=DIARY_ID, 
        username=USERNAME, 
        diary_text=DIARY_TEXT,
        emotion_label=EMOTION,
        image_label=IMAGE
    )
    print("\n💬 상담가의 초기 분석:\n", initial_response)
    print("="*60)

    # 대화 시뮬레이션을 위한 질문 목록
    questions = [
        "제 상황을 어떻게 해결하면 좋을지 막막해요.",
        "혼자 해결하려니 더 힘든 것 같아요. 팀원들에게 어떻게 말하는 게 좋을까요?",
        "좋은 조언이네요. 하지만 제가 버그에 대해 말했을 때 팀원들이 부정적으로 반응할까 봐 걱정돼요.",
        "알겠습니다, 용기를 내볼게요. 그것과 별개로 지금 당장 스트레스를 관리할 수 있는 간단한 방법이 있을까요?"
    ]

    for i, q in enumerate(questions):
        print("\n" + "="*60)
        # 턴 번호는 1부터 시작 (i는 0부터 시작)
        turn_number = i + 1
        print(f"API 요청 {turn_number + 1}: 회원 사용자가 후속 질문 {turn_number}을(를) 보냅니다.")
        print(f"   사용자: {q}")

        # 턴 3 이후 (총 문서 개수가 8개를 넘어서는 시점)부터 요약 기능이 동작할 것을 기대
        if (1 + (turn_number * 2)) >= SUMMARIZATION_TRIGGER_LENGTH:
             print("   (참고: 대화 기록이 길어져서 이번 턴부터 요약 기능이 동작할 수 있습니다.)")
        
        response = continue_counseling_session(
            diary_id=DIARY_ID, 
            username=USERNAME, 
            user_message=q
        )
        print("\n💬 상담가:\n", response)
        print("="*60)


    # --- 시나리오 2: 비회원 사용자 ---
    GUEST_DIARY_TEXT = "오늘 길을 가다가 예쁜 고양이를 봤다. 기분이 좋아졌다."
    
    print("\n" + "="*60)
    print("API 요청 (비회원): 비회원 사용자가 새로운 상담 세션을 시작합니다.")
    guest_initial_response, guest_temp_id = start_new_counseling_session(-1, None, GUEST_DIARY_TEXT, "기쁨", "웃는 고양이")
    print(f"\n💬 상담가의 초기 분석 (임시 ID: {guest_temp_id}):\n", guest_initial_response)
    print("="*60)
    
    # --- 시나리오 3: 기록 삭제 ---
    print("\n" + "="*60)
    print("API 요청 (삭제): 회원 사용자의 대화 기록을 삭제합니다.")
    delete_counseling_history(DIARY_ID, USERNAME)
    delete_counseling_history(-1, guest_temp_id)
    print("="*60)