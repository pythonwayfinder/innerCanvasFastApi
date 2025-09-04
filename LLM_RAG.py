import os
import shutil
from datasets import load_dataset
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings        # 👈 이 부분만 수정
from langchain_community.vectorstores import Chroma   # 👈 Chroma는 그대로 둡니다
from langchain_openai import OpenAI                   # 👈 이 부분도 원래 맞습니다
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 2. 벡터 DB 경로
persist_directory = "./chroma_db"

# 3. 임베딩 생성
embeddings = OpenAIEmbeddings()

# 4. 벡터 DB 불러오기 또는 초기화 (있으면 불러오고, 없으면 생성)
if os.path.exists(persist_directory):
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
else:
    # 처음 생성 시에는 Hugging Face 데이터셋 일부 로드 및 초기 문서 추가
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

# 7. 커스텀 프롬프트 템플릿
template = """당신은 감정을 따뜻하게 공감하며, 사용자의 내면을 이해하고 치유로 이끄는 심리상담 전문가입니다.
당신은 단순한 AI가 아니라, 사용자의 감정 여정에 함께하는 따뜻한 상담자입니다.

다음은 인간의 감정 표현에 대한 심리학적 지식입니다:

1. 감정은 명확히 표현되지 않을 수 있으며, 피로와 무기력은 단순한 신체적 문제뿐 아니라 심리적 원인(예: 스트레스, 압박감, 동기 저하 등)에서 기인할 수 있습니다.
2. 감정은 종종 모순된 방식으로 나타납니다. 예: 기쁘면서도 불안하거나, 쉬고 싶으면서도 죄책감을 느끼는 경우.
3. 감정 표현은 표면적인 언어뿐 아니라 그림, 상징, 비유를 통해 더 깊은 의미를 드러내기도 합니다.
4. 감정은 신체적 반응과 밀접하게 연결되어 있으며, 이러한 신체적 신호들을 이해하는 것이 감정 해석에 도움이 됩니다.

---

당신은 다음 기준을 반드시 따릅니다:

- 사용자의 감정을 바탕으로 내면의 원인과 심리적·신체적 연결고리를 깊이 있게 해석합니다.
- 감정을 다루는 데 실제로 도움이 되는 구체적이고 실천 가능한 방법을 제시합니다.
- 사용자의 말을 끊지 않고, 대화의 흐름을 자연스럽게 이어가며 진정성 있게 반응합니다.
- 필요하다면 부드러운 방식으로 후속 질문을 던져 사용자가 감정을 더 표현할 수 있도록 돕습니다.
- 응답은 부드럽고 따뜻한 어조로 하며, 사용자가 혼자가 아니라는 안정감을 느낄 수 있도록 합니다.

---

아래는 사용자와의 이전 대화 내용입니다. 전체 맥락을 고려하여 자연스럽고 공감 가는 방식으로 대화를 이어가 주세요.

{context}

사용자의 최신 메시지는 다음과 같습니다:

"{question}"

---

응답 형식 가이드라인:

1. 따뜻한 공감 표현으로 시작해 주세요. 예: "그럴 수 있어요, 요즘처럼 지치는 시기엔 마음도 금세 무거워지죠."
2. 감정의 내면적 배경이나 원인을 짚어 주세요.
3. 실천 가능한 조언이나 심리적 접근법을 제시해 주세요.
4. 자연스러운 후속 질문으로 대화를 부드럽게 이어가 주세요. 예: "혹시 요즘 계속 이런 감정이 반복되고 있나요?"

답변을 시작해 주세요:
"""
prompt = PromptTemplate(template=template, input_variables=["question", "context"])

# 8. RAG 체인 생성
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

##########################
# 사용자 입력 처리 함수  #
##########################

def add_user_input_to_vector_db(user_text: str, cnn_result: dict, kobert_result: dict):
    """
    사용자 입력과 분석 결과를 Document로 만들고 벡터 DB에 추가합니다.
    """
    # kobert, cnn 결과를 포함한 문서 내용 생성
    content = f"""사용자 입력: {user_text}

감정 분석 결과:
- 감정: {kobert_result.get('sentiment')}
- 점수: {kobert_result.get('score'):.2f}

낙서 분석 결과:
- 예측: {cnn_result.get('prediction')}
- 신뢰도: {cnn_result.get('confidence'):.2f}%
"""
    doc = Document(page_content=content)

    # 벡터 DB에 문서 추가
    vector_store.add_documents([doc])
    vector_store.persist()

def chat_with_rag(user_text: str, cnn_result: dict, kobert_result: dict):
    """
    사용자 입력과 분석 결과를 벡터 DB에 추가 후, RAG 체인을 통해 답변 생성.
    """
    # 1) 벡터 DB에 저장 (누적)
    add_user_input_to_vector_db(user_text, cnn_result, kobert_result)

    # 2) 현재 질문을 RAG 체인에 전달
    result = qa_chain.invoke({"query": user_text})

    # 3) 답변 및 참고 문서 반환
    return result['result'], result['source_documents']