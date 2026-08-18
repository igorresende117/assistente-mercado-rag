import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Carrega a chave do arquivo .env
load_dotenv()

# 2. Lê os PDFs da pasta 'dados' e fatia o texto
print("Lendo os PDFs...")
arquivos = [
    "dados/challenge01.pdf", 
    "dados/challenge02.pdf", 
    "dados/challenge03.pdf", 
    "dados/challenge04.pdf"
]
documentos = []
for arquivo in arquivos:
    loader = PyPDFLoader(arquivo)
    documentos.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documentos)

# 3. Cria o banco vetorial localmente
print("Criando o banco de dados...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 4. Configura o Gemini com a versão correta
print("Configurando a IA...")
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

rag_chain = (
    {"context": retriever | formatar_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 5. Executa a pergunta de teste
print("\nTudo pronto! Testando...")
pergunta = "O que é o programa Cliente VIP e como faço para me cadastrar?"
resposta = rag_chain.invoke(pergunta)

print(f"\nPergunta: {pergunta}")
print(f"Resposta da IA:\n{resposta}")