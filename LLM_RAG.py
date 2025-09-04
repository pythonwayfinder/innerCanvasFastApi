from dotenv import load_dotenv
import os
from datasets import load_dataset
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma    # <-- 여기를 수정했어요
from langchain_openai import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory


# 2. 벡터 DB 경로
persist_directory = "./chroma_db"

# 3. 임베딩 생성
embeddings = OpenAIEmbeddings(openai_api_key= os.getenv("OPENAI_API_KEY"))

# 4. 벡터 DB 불러오기 또는 초기화
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

# 5. 검색기 생성
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 6. LLM 초기화
llm = OpenAI(temperature=0, max_tokens=1000)

# 질문 요약용 템플릿 (condense_question_prompt)
condense_template = """사용자의 질문을 바탕으로 간단한 대화로 바꿔주세요.
현재 질문: {question}
대화 내역: {chat_history}
요약 질문:"""
condense_prompt = PromptTemplate(
    template=condense_template,
    input_variables=["question", "chat_history"]
)

# 7. 커스텀 프롬프트 템플릿
template = """당신은 감정을 따뜻하게 공감하며, 사용자의 내면을 이해하고 치유로 이끄는 심리상담 전문가입니다.
사용자의 대화 내역과 현재 질문을 바탕으로 공감 어린 답변을 해 주세요.

{context}

사용자의 최신 메시지:
"{question}"

응답을 시작해 주세요:
"""
prompt = PromptTemplate(template=template, input_variables=["question", "context"])

# 8. 메모리 설정 (대화 이력 저장용)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

# 9. ConversationalRetrievalChain 생성 (실시간 대화용)
conv_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    condense_question_prompt=condense_prompt,
    combine_docs_chain_kwargs={"prompt": prompt}
)

##########################
# 사용자 입력 처리 함수  #
##########################

def add_user_input_to_vector_db(user_text: str, cnn_result: dict, kobert_result: dict):
    """
    사용자 입력과 분석 결과를 Document로 만들고 벡터 DB에 추가합니다.
    """
    content = f"""사용자 입력: {user_text}

감정 분석 결과:
- 감정: {kobert_result.get('sentiment')}
- 점수: {kobert_result.get('score'):.2f}

낙서 분석 결과:
- 예측: {cnn_result.get('prediction')}
- 신뢰도: {cnn_result.get('confidence'):.2f}%
"""
    doc = Document(page_content=content)
    vector_store.add_documents([doc])

def chat_with_rag(user_text: str, cnn_result: dict, kobert_result: dict):
    """
    사용자 입력과 분석 결과를 벡터 DB에 추가 후, 대화형 RAG 체인을 통해 답변 생성.
    """
    # 1) 벡터 DB에 저장 (누적)
    add_user_input_to_vector_db(user_text, cnn_result, kobert_result)

    # 2) 현재 질문과 대화 이력을 ConvChain에 전달
    result = conv_chain({"question": user_text})

    # 3) 답변 및 참고 문서 반환
    return result['answer'], result['source_documents']



##########################
# analyze_chat / guest 전용 함수
##########################

def add_user_input_to_vector_db_for_chat(user_text: str, context: dict):
    """
    사용자 입력 + context를 벡터 DB에 저장
    """
    content = f"""사용자 입력: {user_text}

대화 맥락:
- diaryId: {context.get('diaryId')}
- 현재 대화 기록: {context.get('current_chat_history')}
- 지난 7일 대화 기록: {context.get('past_7days_history')}
"""
    doc = Document(page_content=content)

    vector_store.add_documents([doc])
    # 최신 langchain에서는 persist() 필요 없음


def chat_with_rag_for_chat(user_text: str, context: dict):
    """
    analyze_chat / guest 전용:
    사용자 입력 + 컨텍스트를 RAG 체인에 전달하고 답변 생성
    """
    # 1) 벡터 DB에 저장
    add_user_input_to_vector_db_for_chat(user_text, context)

    # 2) RAG 체인 호출
    # conv_chain는 conv_chain({"question": user_text}) 형태로 호출
    # context는 QA용 프롬프트에서 자동 반영되므로 별도로 전달하지 않아도 됨
    result = conv_chain({"question": user_text})

    # 3) AI 답변 및 참고 문서 반환
    return result['answer'], result['source_documents']

# 예시: 대화 시뮬레이션
if __name__ == "__main__":
    while True:
        user_text = input("사용자 입력: ")
        # 여기서는 예시 cnn_result, kobert_result를 임의로 만듦
        cnn_result = {"prediction": "긍정", "confidence": 95.0}
        kobert_result = {"sentiment": "긍정", "score": 0.95}

        answer, sources = chat_with_rag(user_text, cnn_result, kobert_result)
        print("\nAI 응답:")
        print(answer)
        print("\n참고 문서 수:", len(sources))
        print("-" * 50)