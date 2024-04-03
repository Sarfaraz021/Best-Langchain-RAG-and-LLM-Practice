#  ----------RAG Implementation-----------'

from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load PDF files
loader = TextLoader(r"C:\Users\Dell Precision\Desktop\Llama-Langchain-Module\Module-02\prompt.txt", encoding="utf-8")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)
# vectorstore = FAISS.from_texts(texts, embeddings)
vectorstore = Chroma.from_documents(docs, embeddings)


retriever = vectorstore.as_retriever()


template = """
INSTRUCTIONS:

-You are an honest and helpful assistant. Your task is to provide quality responses to the user regarding any related query. 
-If you don't know anything about the question or query asked by the user, do not provide false information; instead, apologize to them.
-Be concise and to the point, addressing the user's specific question. 
-Do not repeat any question asked by the user, nor provide the same response multiple times. 
-Act like a professional chat assistant; if the user praises you, respond with, "It's my pleasure. Do you have any other questions? I   would be happy to assist."

REMEMBER:
Only answer from this context {context} when user query or question is related to this context otherwise behave like a normal chatbot to answer the user according to their query or question.

{question}:
Answer : 

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
    question = input("User>: ")
    if question.lower() == "exit":
        print("Great to chat with you! By By.")
        break
    else:
        response  = chain.invoke(question)

        for chunks in chain.stream(response):
            print(chunks, end="", flush=True)

        print("Assitant>:" +response)
        print("******************************************************************\n")
        