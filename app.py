import streamlit as st
import tempfile
import string
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain.document_loaders import TextLoader, PyPDFLoader
from huggingface_hub import login


# ------------------- Streamlit Page Config -------------------
st.set_page_config(
    page_title="ChatPDF",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("📑 Document Q&A Chatbot")
st.sidebar.write("Upload a PDF or TXT file, and then ask questions about it!")

# ------------------- File Upload -------------------
uploaded_file = st.sidebar.file_uploader("Upload your file", type=["txt", "pdf"], key="upload_file")

# ------------------- Hugging Face Login -------------------
key_token = st.sidebar.text_input("🔑 Hugging Face Token", placeholder="Enter your HF token here")
if key_token:
    try:
        st.session_state['login'] = key_token
        login(st.session_state['login'])
        st.sidebar.success("Logged in successfully ✅")
    except ValueError:
        st.sidebar.error("Invalid token ❌")


# ------------------- Document Processing -------------------
if uploaded_file:
    # Reset when a new file is uploaded
    if "last_uploaded" not in st.session_state or st.session_state["last_uploaded"] != uploaded_file.name:
        st.session_state["messages"] = []
        st.session_state["chat_objects"] = []
        st.session_state["last_uploaded"] = uploaded_file.name
        st.session_state["retriever"] = None  # reset retriever as well

    # Save uploaded file temporarily
    suffix = ".txt" if uploaded_file.type == "text/plain" else ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    # Load documents
    loader = TextLoader(tmp_file_path) if uploaded_file.type == "text/plain" else PyPDFLoader(tmp_file_path)
    docs = loader.load()
    st.sidebar.success("✅ Document loaded successfully!")

    # --- clean + split text ---
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

    def clean_str(text: str) -> str:
        allowed_punctuations = ".,?()[]{}*%"
        allowed = string.ascii_letters + string.digits + string.whitespace + allowed_punctuations
        table = str.maketrans("", "", "".join(set(map(chr, range(0x110000))) - set(allowed)))
        cleaned = text.translate(table)
        return " ".join(cleaned.split())

    texts = []
    for d in docs:
        cleaned = clean_str(d.page_content)
        texts.extend(splitter.split_text(cleaned))

    docs = [Document(page_content=text) for text in texts]

    # ------------------- Embeddings (cached) -------------------
    @st.cache_resource
    def load_embeddings():
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    embedding_func = load_embeddings()

    # ------------------- Vector Store (NOT cached) -------------------
    if st.session_state.get("retriever") is None:
        vector_store = FAISS.from_documents(docs, embedding_func)
        st.session_state["retriever"] = vector_store.as_retriever()

    retriever = st.session_state["retriever"]

    def text_extractor(docs):
        return " ".join(doc.page_content for doc in docs)

    # ------------------- LLM + Prompt -------------------
    prompt = ChatPromptTemplate([
        (
            "system",
            "You are an AI assistant that answers user questions from the given context. "
            "If the question is out of scope, just say 'The question is out of topic'. "
            "You can also reply to general chat messages naturally.\n\n"
            "FILE CONTEXT: {context}"
        ),
        *st.session_state["chat_objects"],
        ("human", "{question}")
    ])

    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        task="text-generation"
    )

    model = ChatHuggingFace(llm=llm)
    parser = StrOutputParser()

    # ------------------- Chat UI -------------------
    st.title("🤖 Chat with your Document")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "chat_objects" not in st.session_state:
        st.session_state["chat_objects"] = []

    # Display chat history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt_ := st.chat_input("Type your question..."):
        # Save user msg
        st.session_state["messages"].append({"role": "user", "content": prompt_})
        st.session_state["chat_objects"].append(HumanMessage(content=prompt_))
        with st.chat_message("user"):
            st.markdown(prompt_)

        # Chain
        parallel_chain = RunnableParallel({
            "context": RunnableLambda(lambda x: retriever.invoke(x["question"])) | RunnableLambda(text_extractor),
            "question": RunnablePassthrough(),
            "chats": RunnablePassthrough()
        })

        final_chain = parallel_chain | prompt | model | parser
        result = final_chain.invoke({"chats": st.session_state["chat_objects"], "question": prompt_})

        st.session_state["chat_objects"].append(AIMessage(content=result))

        # AI response
        st.session_state["messages"].append({"role": "assistant", "content": result})
        with st.chat_message("assistant"):
            st.markdown(result)
