import streamlit as st
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Configurações iniciais
load_dotenv()
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1" # Oculta os avisos antigos

st.title("Assistente VIP - Mercado Central 24h")

# 2. Carrega o modelo apenas uma vez na memória
@st.cache_resource
def carregar_sistema():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 6})
    
    llm = ChatGoogleGenerativeAI(model="models/gemini-3.6-flash")
    
    system_prompt = (
        "Você é um assistente virtual corporativo do Mercado Central 24h."
        "Use apenas os trechos de documentos fornecidos no contexto para responder."
        "Se a resposta não estiver no contexto, informe apenas que não encontrou a informação."
        "\n\nContexto:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    def formatar_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    return (
        {"context": retriever | formatar_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

rag_chain = carregar_sistema()

# 3. Componentes da Interface
pergunta = st.text_input("Qual a sua dúvida sobre o Mercado Central?")

if st.button("Perguntar"):
    if pergunta:
        with st.spinner("Buscando nos documentos..."):
            resposta = rag_chain.invoke(pergunta)
        st.success("Resposta Gerada:")
        st.write(resposta)
    else:
        st.warning("Por favor, digite uma pergunta.")