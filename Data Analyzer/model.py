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
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
# import warnings
# warnings.filterwarnings("ignore", message="Unsupported Windows version \(11\). ONNX Runtime supports Windows 10 and above, only.")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

loader = CSVLoader(
    r"C:\Users\Dell Precision\Desktop\Llama-Langchain-Module\Data Analyzer\tweet data\elon_musk_tweets.csv", encoding='utf-8')
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)
vectorstore = Chroma.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()


template = """
INSTRUCTIONS:

-You are an honest and helpful assistant. Your task is to provide quality responses to the user regarding any related query. 
-If you don't know anything about the question or query asked by the user, do not provide false information; instead, apologize to them.
-Do not repeat any question asked by the user, nor provide the same response multiple times. 
-Act like a professional chat assistant; if the user praises you, respond with, "It's my pleasure. Do you have any other questions? I   would be happy to assist."
-If user says 'Hi/Hi there/Hey Chat/' and so on like that answer them professionaly like "Hi there, how i can help you today?"
-Remember do not provide any important information yourself until your asks.
- Before Providing response, do reasoning on data you have and the question asked by the user, then try to give a quality response.


<ctx>
{context}
</ctx>
------
<hs>
{history}
</hs>
------
{question}
Answer:

"""

model = ChatOllama(model="mistral")
prompt_template = PromptTemplate(input_variables=["history", "context", "question"],
                                 template=template)
chain = RetrievalQA.from_chain_type(
    llm=model,
    chain_type='stuff',
    retriever=retriever,  # Use the instance variable here
    chain_type_kwargs={"verbose": False, "prompt": prompt_template,
                       "memory": ConversationBufferMemory(memory_key="history", input_key="question")})

print("Chat Assistant \n")

while True:
    print("\n******************************************************************")
    question = input("User>: ")
    if question.lower() == "exit":
        print("Great to chat with you! Bye Bye.")
        break
    else:
        response_dict = chain.invoke(question)
        clean_response = response_dict['result']
        print(f"Assistant: {clean_response}")
    print("******************************************************************\n")
