from fastapi import FastAPI
from pydantic import BaseModel
from langchain_chroma.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

CAMINHO_DB = "db"

app = FastAPI()

# Liberar requisições do front-end
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PerguntaRequest(BaseModel):
    pergunta: str


prompt_template = """
Responda a pergunta do usuario:
{pergunta}

com base nessas informações abaixo:
{base_conhecimento} 
"""


@app.post("/perguntar")
def perguntar(req: PerguntaRequest):
    pergunta = req.pergunta

    funcao_embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = Chroma(persist_directory=CAMINHO_DB, embedding_function=funcao_embedding)

    resultados = db.similarity_search_with_relevance_scores(pergunta, k=4)
    if len(resultados) == 0 or resultados[0][1] < 0.6:
        return {"resposta": "Não encontrou informação relevante na base"}

    textos_resultado = [r[0].page_content for r in resultados]
    base_conhecimento = "\n\n----\n\n".join(textos_resultado)

    prompt = ChatPromptTemplate.from_template(prompt_template)
    prompt_final = prompt.invoke({"pergunta": pergunta, "base_conhecimento": base_conhecimento})

    modelo = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    resposta = modelo.invoke(prompt_final).content

    return {"resposta": resposta}
