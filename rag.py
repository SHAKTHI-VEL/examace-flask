from flask import Flask,request,jsonify
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
from flask_cors import CORS, cross_origin
import os

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1",encode_kwargs = {'precision': 'binary'})

# Vector Store
one_bit_vectorstore = FAISS.load_local("CN-dataStore", embeddings, allow_dangerous_deserialization=True)
retriever = one_bit_vectorstore.as_retriever()

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route("/", methods = ['POST'])
def hello_world():
        content = request.json

        # embeddings
        HF_TOKEN = os.getenv("HF_TOKEN")

        #llm
        llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.3" ,huggingfacehub_api_token=HF_TOKEN)

        #template
        QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI  model teaching assistant. Your task is to generate  best versions of the given user question to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, your goal is to answer the question correctly and help the user overcome some of the limitations of the distance-based similarity search. Provide these alternative questions separated by newlines""",)

        template = """Answer the question based ONLY on the following context:
        {context}
        Question: {question}
        """

        # Prompt
        prompt = ChatPromptTemplate.from_template(template)

        chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
        )

        #response
        response = chain.invoke(content['question'])

        return {"answer":response}
        
        