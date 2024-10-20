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
from pptx import Presentation
from pptx.util import Inches
import os

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1",encode_kwargs = {'precision': 'binary'})

# Vector Store
one_bit_vectorstore = FAISS.load_local("CN-dataStore", embeddings, allow_dangerous_deserialization=True)
retriever = one_bit_vectorstore.as_retriever()

# one_bit_vectorstore_ai = FAISS.load_local("AI-dataStore", embeddings, allow_dangerous_deserialization=True)
# retriever_ai = one_bit_vectorstore_ai.as_retriever()


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

def create_slide(presentation, title, content_points):
    slide_layout = presentation.slide_layouts[1]  # Use the layout with title and content
    slide = presentation.slides.add_slide(slide_layout)
    
    title_placeholder = slide.shapes.title
    content_placeholder = slide.placeholders[1]
    
    title_placeholder.text = title
    
    for point in content_points:
        p = content_placeholder.text_frame.add_paragraph()
        p.text = point
        p.level = 0  # Bullet point level

def generate_presentation(slide_data, template_path, output_path):
    presentation = Presentation(template_path)
    
    for slide_info in slide_data:
        title = slide_info['title']
        content_points = slide_info['content']
        
        while content_points:
            slide = presentation.slides.add_slide(presentation.slide_layouts[1])
            title_placeholder = slide.shapes.title
            content_placeholder = slide.placeholders[1]
            
            title_placeholder.text = title
            
            # Add points to the slide until it is full
            remaining_points = []
            for point in content_points:
                p = content_placeholder.text_frame.add_paragraph()
                p.text = point
                p.level = 0  # Bullet point level
                
                # Check if the content exceeds the slide's capacity
                if content_placeholder.text_frame.fit_text():
                    remaining_points.append(point)
                    content_placeholder.text_frame.text = content_placeholder.text_frame.text.rsplit('\n', 1)[0]
                    break
            
            content_points = remaining_points
    
    presentation.save(output_path)


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

@app.route('/ppt', methods = ['GET'])
def ppt():
        slides_data = [
    {
        "title": "Introduction to AI",
        "content": [
            "Definition of Artificial Intelligence",
            "History of AI development",
            "Importance of AI in today's world",
            "Applications in various industries"
        ]
    },
    {
        "title": "Types of AI",
        "content": [
            "Narrow AI",
            "General AI",
            "Superintelligent AI",
            "Examples of Narrow AI in use"
        ]
    },
    {
        "title": "AI and Machine Learning",
        "content": [
            "Overview of Machine Learning",
            "Difference between AI and ML",
            "Types of Machine Learning: Supervised, Unsupervised, Reinforcement",
            "Real-world applications of Machine Learning"
        ]
    },
    {
        "title": "Challenges in AI",
        "content": [
            "Ethical concerns",
            "Bias in AI models",
            "Data privacy issues",
            "Transparency and explainability"
        ]
    },
    {
        "title": "Future of AI",
        "content": [
            "Predictions for AI advancements in a few years time due to the rapid growth of technology is that AI will be able to perform tasks that are currently impossible for machines to do also AI will be able to perform tasks that are currently impossible for machines to do", 
            "Predictions for AI advancements in a few years time due to the rapid growth of technology is that AI will be able to perform tasks that are currently impossible for machines to do also AI will be able to perform tasks that are currently impossible for machines to do. AI in healthcare, finance, and education. Predictions for AI advancements in a few years time due to the rapid growth of technology is that AI will be able to perform tasks that are currently impossible for machines to do also AI will be able to perform tasks that are currently impossible for machines to do. AI in healthcare, finance, and education.",
            "Impact on job markets",
            "Opportunities for future research"
        ]
    }
]

        template_path = 'template.pptx'  # Path to your PowerPoint template
        output_path = 'output.pptx'  # Path to save the generated presentation

        generate_presentation(slides_data, template_path, output_path)
        return {"success":"true","ppt_path":"output.pptx"}



# @app.route("/ai", methods = ['POST'])
# def ai():
#         content = request.json

#         # embeddings
#         HF_TOKEN = os.getenv("HF_TOKEN")

#         #llm
#         llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.3" ,huggingfacehub_api_token=HF_TOKEN)

#         #template
#         QUERY_PROMPT = PromptTemplate(
#         input_variables=["question"],
#         template="""You are an AI  model teaching assistant. Your task is to generate  best versions of the given user question to retrieve relevant documents from a vector database. By generating multiple perspectives on the user question, your goal is to answer the question correctly and help the user overcome some of the limitations of the distance-based similarity search. Provide these alternative questions separated by newlines""",)

#         template = """Answer the question based ONLY on the following context:
#         {context}
#         Question: {question}
#         """

#         # Prompt
#         prompt = ChatPromptTemplate.from_template(template)

#         chain = (
#         {"context": retriever_ai, "question": RunnablePassthrough()}
#         | prompt
#         | llm
#         | StrOutputParser()
#         )

#         #response
#         response = chain.invoke(content['question'])

#         return {"answer":response}

if __name__ == '__main__':
    app.run(debug=True,host= '0.0.0.0',port=3000)