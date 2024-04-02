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


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

loader = TextLoader(
    r"C:\Users\Dell Precision\Desktop\Llama-Langchain-Module\Module-02\prompt.txt", encoding="utf-8")
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
-Be concise and to the point, addressing the user's specific question. 
-Do not repeat any question asked by the user, nor provide the same response multiple times. 
-Act like a professional chat assistant; if the user praises you, respond with, "It's my pleasure. Do you have any other questions? I   would be happy to assist."

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
                       "memory": ConversationBufferMemory(memory_key="history", input_key="question")}
)


print("Chat Assistant \n")

while True:
    print("\n******************************************************************")
    question = input("User>: ")
    if question.lower() == "exit":
        print("Great to chat with you! By By.")
        break
    else:
        response_dict = chain.invoke(question)
        clean_response = response_dict['result']
        print(f"Assistant: {clean_response}")
        print("******************************************************************\n")
