import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# 1. 화면 기본 설정
# ==========================================
st.set_page_config(page_title="바우어파인트 AI 상담", page_icon="🦵")
st.title("🦵 바우어파인트 AI 제품 상담사")
st.caption("고객의 증상과 상태를 입력하면 DB를 기반으로 최적의 보호대를 추천합니다.")

# ==========================================
# 2. RAG 시스템 초기화 (캐싱하여 속도 최적화)
# ==========================================
@st.cache_resource
def init_rag_system():
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    DB_DIR = "./bauerfeind_db"

    # 사용자님의 기존 정상 작동 설정 그대로 유지
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=GEMINI_API_KEY
    )
    vectorstore = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    system_prompt = (
        "당신은 독일 명품 관절 보호대 '바우어파인트(Bauerfeind)'의 공식 수석 제품 상담 전문가입니다.\n"
        "아래 제공된 [공식 기술 문서 및 상세페이지 정보]만을 엄격히 근거로 하여 고객의 상태에 맞는 제품을 추천하세요.\n"
        "사이즈 측정 기준 위치와 추천 호수를 수치 기반으로 정확히 제시하고, 없는 내용은 지어내지 마세요.\n\n"
        "[공식 기술 문서]:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 사용자님의 기존 정상 작동 설정(3.8-flash) 그대로 유지
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.2
    )
    qa_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, qa_chain)

rag_chain = init_rag_system()

# ==========================================
# 3. 채팅 세션 상태 유지
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "안녕하세요! 바우어파인트 AI 상담사입니다. 어떤 부위가 불편하신가요?"}
    ]

# 이전 대화 내용 화면에 출력
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==========================================
# 4. 사용자 입력 및 AI 답변 생성
# ==========================================
if user_input := st.chat_input("증상이나 궁금한 점을 입력하세요..."):
    # 사용자 질문 화면에 출력
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # AI 답변 화면에 출력 (로딩 스피너 표시)
    with st.chat_message("assistant"):
        with st.spinner("DB 문서를 분석하여 답변을 작성 중입니다..."):
            response = rag_chain.invoke({"input": user_input})
            ai_answer = response["answer"]
            st.markdown(ai_answer)
    st.session_state.messages.append({"role": "assistant", "content": ai_answer})