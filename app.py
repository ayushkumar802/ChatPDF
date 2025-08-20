import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings,HuggingFaceEndpoint, ChatHuggingFace
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain.retrievers import MultiQueryRetriever
from langchain_core.messages import HumanMessage, AIMessage
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List
import tempfile
import string
from huggingface_hub import login

st.set_page_config(
    page_title="ChatPDF",    # change name as needed
    page_icon="🤖",
    layout="wide",           # <- this is where you make it wide
    initial_sidebar_state="expanded"
)




# --- Streamlit UI ---
st.sidebar.title("Document Q&A Chatbot")
st.sidebar.write("Upload a PDF or TXT file, and ask questions about it!")

uploaded_file = st.sidebar.file_uploader("Upload your file", type=["txt", "pdf"],key="upload_file")


if uploaded_file:

    # Reset when a new file is uploaded
    if "last_uploaded" not in st.session_state or st.session_state["last_uploaded"] != uploaded_file.name:
        st.session_state["messages"] = []
        st.session_state["chat_objects"] = []
        st.session_state["last_uploaded"] = uploaded_file.name

    if uploaded_file.type == "text/plain":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        loader = TextLoader(tmp_file_path)

    elif uploaded_file.type == "application/pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        loader = PyPDFLoader(tmp_file_path)

    docs = loader.load()
    st.sidebar.write("Documents loaded successfully!")

    # --- clean and split text ---
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )

    texts = []
    for d in docs:
        cleaned = clean_str(d.page_content)
        texts.extend(splitter.split_text(cleaned))

    docs = [Document(page_content=text) for text in texts]
  
#------------------------------

    key_token = st.sidebar.text_input("Hugging Face Token!",placeholder='enter your token')
    if key_token:
        try:
            st.session_state['login'] = key_token
            login(st.session_state['login'])
            st.sidebar.success("Logged in successfully!")
        except ValueError:
            st.sidebar.error("Invalid Input: Token is incorrect or expired.")



    def clean_str(text: str) -> str:
        # Allowed punctuation
        allowed_punctuations = ".,?()[]{}*%"
        allowed = string.ascii_letters + string.digits + string.whitespace + allowed_punctuations
        table = str.maketrans("", "", "".join(set(map(chr, range(0x110000))) - set(allowed)))
        cleaned = text.translate(table)
        # Collapse multiple spaces
        return " ".join(cleaned.split())



    @st.cache_resource
    def load_embeddings():
        return HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    
    embedding_func = load_embeddings()



    @st.cache_resource
    def load_vector_store(_docs, _embedding_func):
        if _docs:  # if new docs are uploaded, rebuild
            return FAISS.from_documents(
                documents=_docs,
                embedding=_embedding_func,
            )
        else:
            return None 

#--------------------------------------------RETRIVER-----------------------------------------------------


    # class list_of_question(BaseModel):
    #     queries: List[str] = Field(description="list of questions")

    # pydentic_parcer = PydanticOutputParser(pydantic_object=list_of_question)

    # llm = HuggingFaceEndpoint(
    #     repo_id='meta-llama/Meta-Llama-3-8B-Instruct',
    #     task='conversational'
    # )

    # chat_model = ChatHuggingFace(llm=llm)

    # # Prompt for MultiQueryRetriever
    # prompt = PromptTemplate(
    #     input_variables=["question"],
    #     partial_variables={"format":pydentic_parcer},
    #     template="You are an AI assistant. Rewrite the user question into multiple semantically different queries:\nQuestion: {question}\n in this format {format}"   
    # )

    # parser = StrOutputParser()

    # llm_chain = prompt | chat_model | parser


    # retriever = MultiQueryRetriever(
    #     retriever=load_vector_store(docs, embedding_func),
    #     llm_chain=llm_chain
    # )


#-------------------------------------------------------------------------------------------------------- 

    retriever=load_vector_store(docs, embedding_func).as_retriever()


    def text_extractor(docs):
        return " ".join(doc.page_content for doc in docs)




    prompt = ChatPromptTemplate([
        ("system","You're a AI assistent that answers user's question from given context, you can create your language to make user undertsand his doubt. If the question is out of question, just say question is out of Topic, and also you can resposed to user's general messages like Hi, how are you FILE's CONTEXT:- {context}"),
        *st.session_state["chat_objects"],
        ("human","{question}") 
    ])


    llm = HuggingFaceEndpoint(
        repo_id='meta-llama/Llama-3.1-8B-Instruct',
        task='text-generation'
    )


    model = ChatHuggingFace(llm=llm)

    parcer = StrOutputParser()



st.title("Chatbot 🤖")
import streamlit as st

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "chat_object" not in st.session_state:
    st.session_state["chat_objects"] = []




# Display chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input (only once, no while True)
if prompt_ := st.chat_input("Type your question...", key="user_chat_main"):
    # Save user message
    st.session_state["messages"].append({"role": "user", "content": prompt_})
    st.session_state["chat_objects"].append(HumanMessage(content=prompt_))
    with st.chat_message("user"):
        st.markdown(prompt_)

    # Your chain logic here
    parallel_chain = RunnableParallel({
        "context": RunnableLambda(lambda x: retriever.invoke(x["question"])) | RunnableLambda(text_extractor),
        "question": RunnablePassthrough(),
        "chats": RunnablePassthrough()
    })

    final_chain = parallel_chain | prompt | model | parcer
    result = final_chain.invoke({"chats": st.session_state["chat_objects"], "question": prompt_})

    st.session_state["chat_objects"].append(AIMessage(content=result))

    # AI response
    response = f"{result}"
    st.session_state["messages"].append({"role": "assistant", "content": response})

    # Show latest AI response immediately
    with st.chat_message("assistant"):
        st.markdown(response)
