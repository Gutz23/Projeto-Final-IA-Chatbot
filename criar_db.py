from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os


load_dotenv()

PASTA_BASE = "base"

def criar_db():
    documentos = carregar_documentos()
    chunks = dividir_chunks(documentos)
    vetorizar_chunks(chunks)

    #chunks = dividir_chunks(documentos)
    #vetorizar_chunks


def carregar_documentos():
    carregador = PyPDFDirectoryLoader(PASTA_BASE, glob="*.pdf")
    documentos = carregador.load()
    return documentos

def dividir_chunks(documentos):
    separador_documentos = RecursiveCharacterTextSplitter(
        chunk_size=10500,
        chunk_overlap=2000,
        length_function=len,
        add_start_index=True
    )
    chunks = separador_documentos.split_documents(documentos)
    print(len(chunks))
    return chunks

def vetorizar_chunks(chunks):
    # Verifica se a chave está definida
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("A variável de ambiente GOOGLE_API_KEY não está definida no .env")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = Chroma.from_documents(chunks, embeddings, persist_directory="db")
    print("Banco de dados vetorial criado com sucesso.")

criar_db()