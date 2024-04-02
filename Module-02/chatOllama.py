# # LangChain supports many other chat models. Here, we're using Ollama
# from langchain_community.chat_models import ChatOllama
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate

# # supports many more optional parameters. Hover on your `ChatOllama(...)`
# # class to view the latest available supported parameters
# llm = ChatOllama(model="mistral")
# template = """
# INSTRUCTIONS:

# -You are an honest and helpful assistant. Your task is to provide quality responses to the user regarding any related query. 
# -If you don't know anything about the question or query asked by the user, do not provide false information; instead, apologize to them.
# -Be concise and to the point, addressing the user's specific question. 
# -Do not repeat any question asked by the user, nor provide the same response multiple times. 
# -Act like a professional chat assistant; if the user praises you, respond with, "It's my pleasure. Do you have any other questions? I   would be happy to assist."



# {user_input}
# Assistant: " "

# """
# print("Chat Assistant \n")

# while True:
#     print("\n******************************************************************")
#     user_input = input("User>: ")
#     if user_input.lower() == "exit":
#         print("Great to chat with you! By By.")
#         break
#     else:
#         prompt = ChatPromptTemplate.from_template(template)
#         chain = prompt | llm | StrOutputParser()
#         response  = chain.invoke({"user_input": user_input})

#         for chunks in chain.stream(response):
#             print(chunks, end="", flush=True)

#         print("Assitant>:" +response)
#         print("******************************************************************\n")
        



# ----------RAG Implementation-----------'

from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
import os
from dotenv import load_dotenv
load_dotenv("var.env")

os.environ["OPENAI_API_KEY"]

# pip install sentence-transformers
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embeddings=OpenAIEmbeddings()


# Load PDF files
loader = TextLoader("C:\Users\Dell Precision\Desktop\Llama-Langchain-Module\prompt.txt", encoding="utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)
vectorstore = FAISS.from_texts(texts, embeddings)
retriever = vectorstore.as_retriever()


template = """
INSTRUCTIONS:

-You are an honest and helpful assistant. Your task is to provide quality responses to the user regarding any related query. 
-If you don't know anything about the question or query asked by the user, do not provide false information; instead, apologize to them.
-Be concise and to the point, addressing the user's specific question. 
-Do not repeat any question asked by the user, nor provide the same response multiple times. 
-Act like a professional chat assistant; if the user praises you, respond with, "It's my pleasure. Do you have any other questions? I   would be happy to assist."

Answer the question based only on the following context:
{context}

Question: {question}
"""
model = ChatOllama(model="mistral")
prompt = ChatPromptTemplate.from_template(template)
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)


print("Chat Assistant \n")

while True:
    print("\n******************************************************************")
    user_input = input("User>: ")
    if user_input.lower() == "exit":
        print("Great to chat with you! By By.")
        break
    else:
        response  = chain.invoke(user_input)

        for chunks in chain.stream(response):
            print(chunks, end="", flush=True)

        print("Assitant>:" +response)
        print("******************************************************************\n")
        