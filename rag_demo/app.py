import streamlit as st

from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------

st.title("RAG Application")


# --------------------------------------------------
# URLs
# --------------------------------------------------

urls = [
    "https://www.victoriaonmove.com.au/local-removalists.html",
    "https://victoriaonmove.com.au/index.html",
    "https://victoriaonmove.com.au/contact.html"
]


# --------------------------------------------------
# Load Web Pages
# --------------------------------------------------

@st.cache_resource
def load_documents():

    loader = UnstructuredURLLoader(urls=urls)

    data = loader.load()

    return data


data = load_documents()


# --------------------------------------------------
# Split Documents
# --------------------------------------------------

@st.cache_resource
def split_documents(data):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = text_splitter.split_documents(data)

    return docs


docs = split_documents(data)


# --------------------------------------------------
# Hugging Face Embeddings
# --------------------------------------------------

@st.cache_resource
def create_vectorstore(docs):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings
    )

    return vectorstore


vectorstore = create_vectorstore(docs)


# --------------------------------------------------
# Retriever
# --------------------------------------------------

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6}
)


# --------------------------------------------------
# Hugging Face LLM
# --------------------------------------------------

@st.cache_resource
def create_llm():

    llm = HuggingFacePipeline.from_model_id(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 200,
            "temperature": 0.4,
            "do_sample": True
        }
    )

    return llm


llm = create_llm()


# --------------------------------------------------
# RAG Prompt
# --------------------------------------------------

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following retrieved context to answer the question. "
    "If you don't know the answer, say that you don't know. "
    "Keep the answer concise and use a maximum of three sentences."
    "\n\n"
    "Context:\n{context}"
)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)


# --------------------------------------------------
# Format Retrieved Documents
# --------------------------------------------------

def format_docs(docs):

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


# --------------------------------------------------
# RAG Chain
# --------------------------------------------------

rag_chain = (
    {
        "context": retriever | format_docs,
        "input": RunnablePassthrough()
    }
    | prompt
    | llm
)


# --------------------------------------------------
# Chat Input
# --------------------------------------------------

query = st.chat_input("Ask a question about Victoria On Move")


if query:

    response = rag_chain.invoke(query)

    st.write(response)