import os
import requests
import streamlit as st
from langchain.llms import Together
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from duckduckgo_search import DDGS
from langchain.chat_models import ChatOpenAI
from langfuse import Langfuse
from langfuse.callback import CallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate, HumanMessagePromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch, RunnablePassthrough
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from googlesearch import search

load_dotenv()

langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
api_key = os.getenv("api_key")

langfuse = Langfuse()
handler = CallbackHandler(langfuse_public_key, langfuse_secret_key, "https://cloud.langfuse.com")



class Model:
    def __init__(self):
        self.together_token = os.getenv('api_key')
        self.langfuse_sk = os.getenv('LANGFUSE_SECRET_KEY')
        self.langfuse_pk = os.getenv('LANGFUSE_PUBLIC_KEY')
        self.model_name = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
        self.base_url = 'https://api.together.xyz'
        self.temperature = 0
        self.streaming = False
        self.langfuse_callbak = CallbackHandler(
            self.langfuse_pk, self.langfuse_sk)

    def get_model(self):

        model = ChatOpenAI(
            base_url=self.base_url,
            model=self.model_name,
            api_key=self.together_token,
            streaming=self.streaming,
            temperature=self.temperature,
            callbacks=[self.langfuse_callbak]
        )

        return model

class DuckDuckGoSearch:
    def __init__(self):
        self.base_url = "https://api.duckduckgo.com/"

    def search(self, query, max_results=5):
        params = {
            "q": query,
            "format": "json",
            "no_redirect": 1,
            "no_html": 1,
            "skip_disambig": 1,
            "t": "streamco",
            "kl": "es-ES"
        }
        response = requests.get(self.base_url, params=params)
        data = response.json()
        results = []
        for i in range(min(max_results, len(data['RelatedTopics']))):
            results.append(data['RelatedTopics'][i]['Text'])
        return results


class Tool:
    def __init__(self, name, description, func):
        self.name = name
        self.description = description
        self.func = func



# Interfaz de usuario con Streamlit
option = st.radio("Seleccione una opción:", ("Búsqueda en internet", "Que mire en el RAG"))

resultado = None  # Definir resultado como None por defecto
question = None  # Definir question como None por defecto

if option == "Búsqueda en internet":
    question = st.text_input("Introduzca su pregunta, señor: ", key="busqueda_input")
    if question:
        
        google_search= search(question,advanced=True,num_results=5,lang='en')
 
        search_results=[]

        for result in google_search:

                search_results.append({
                    'title':result.title,
                    'url':result.url,
                    'description':result.description,
        })
        resultado = search_results
        # Crear un mensaje del sistema con la descripción del asistente
        system_message = SystemMessage(
            content=(
                "Eres un experto en ciberseguridad capaz de responder preguntas de manera concisa y precisa. "
                "Seleccionaste la opción de búsqueda en internet. "
                "Te voy a dejar una búsqueda en internet, para mejorar la respuesta del usuario recibirás un array de diccionarios, estos diccionarios contienen las siguientes claves y valores:"

            """{
            'title':'web title',
            'url':'web url',
            'description':'description of the web',
            }
"""
                f"El resultado de la búsqueda en internet es: {str(resultado)}" if resultado is not None else "No se encontró información relevante en la opción seleccionada."
            )
        )
elif option == "Que mire en el RAG":
    question = st.text_input("Introduzca el riesgo que desea consultar: ", key="riesgo_input")
    markdown_files = ["C:/Users/Jose Antonio/DocumentosBigData/MIA/ProyectoMinimo/playbook1.md"]

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    documents = []

    for file in markdown_files:
        with open(file, 'r') as f:
            markdown_string = f.read()
        doc = markdown_splitter.split_text(markdown_string)
        documents += doc


    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])


    # file_paths = [
    #    "ruta_del_archivo1.md",
    #    "ruta_del_archivo2.md",
    #    "ruta_del_archivo3.md"
    # ]


    embedding_function = SentenceTransformerEmbeddings(model_name="BAAI/bge-large-en")

    db = Chroma.from_documents(documents, embedding_function, collection_metadata={
                                         "hnsw:space": "cosine"}, persist_directory="./db")

    load_vector_store = Chroma(
        persist_directory="./db", embedding_function=embedding_function)
    retriever = load_vector_store.as_retriever(search_kwargs={"k": 1})

    docs = db.similarity_search(question)

    for doc in docs:
        contenidomd = (doc.page_content)

                # Crear un mensaje del sistema con la descripción del asistente
    system_message = SystemMessage(
                    content=(
                        "Eres un experto en ciberseguridad capaz de responder preguntas de manera concisa y precisa. "
                        "Seleccionaste la opción de consultar en el RAG. "
                        f"El resultado del análisis del RAG es: {contenidomd}" if contenidomd is not None else "No se encontró información relevante en el RAG."
                    )
                )

# Crear una plantilla de mensaje humano para la pregunta del usuario
if question:
    human_message = HumanMessagePromptTemplate.from_template("{question}")
else:
    human_message = None

# Crear la plantilla de chat combinando el mensaje del sistema y el mensaje human
if human_message:
    # Formatear los mensajes con la pregunta del usuario
    #messages = chat_template.format_messages(question=question)

    
    
    


   
    model=Model()
    llm=model.get_model()
    prompt_messages = [
            SystemMessage(content=str(system_message)),
            HumanMessage(content=str(question))
        ]

    # Ejecutar el modelo de lenguaje con la pregunta del usuario
    result = llm.invoke(prompt_messages)

    # Mostrar la respuesta al usuario
    st.write('Respuesta: ', result.content)
else:
    st.write("Seleccione una opción válida")


# Llamada a la api para recoger ultima alerta y explicar