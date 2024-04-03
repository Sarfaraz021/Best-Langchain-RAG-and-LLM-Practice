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
# import warnings
# warnings.filterwarnings("ignore", message="Unsupported Windows version \(11\). ONNX Runtime supports Windows 10 and above, only.")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

loader = DirectoryLoader(
    r"C:\Users\Dell Precision\Desktop\Llama-Langchain-Module\Module-02\data2")
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

template_recipes = """
INSTRUCTIONS:
Provide a list of top 10 dishes from the context you have.
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

template_recipes_instructions = """
INSTRUCTIONS:
Provide a detailed instructions and ingrdiants to make required to make the dish.
Remember be detailed, and use bullet points and and sub bullet points to make this dish
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
prompt_template_recipes = PromptTemplate(input_variables=["history", "context", "question"],
                                 template=template_recipes)

prompt_template_recipes_instructions = PromptTemplate(input_variables=["history", "context", "question"],
                                 template=template_recipes_instructions)

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

chain_recipes = RetrievalQA.from_chain_type(
    llm=model,
    chain_type='stuff',
    retriever=retriever,  
    chain_type_kwargs={"verbose": False, "prompt": prompt_template_recipes,
                       "memory": ConversationBufferMemory(memory_key="history", input_key="question")}
)

chain_recipes_instructions = RetrievalQA.from_chain_type(
    llm=model,
    chain_type='stuff',
    retriever=retriever,  
    chain_type_kwargs={"verbose": False, "prompt": prompt_template_recipes_instructions,
                       "memory": ConversationBufferMemory(memory_key="history", input_key="question")}
)


print("Chat Assistant \n")

# Initialize an empty list to store dishes
dishes = []

while True:
    print("\n******************************************************************")
    question = input("User>: ")
    if question.lower() == "exit":
        print("Great to chat with you! Bye Bye.")
        break
    elif question.lower() in ('recipes', 'recipe'):
        response_dict = chain_recipes.invoke(question)
        clean_response = response_dict['result']
        print(f"Assistant: {clean_response}")
        
        dishes = clean_response.split('\n')[2:]  
    elif question.isdigit() and 1 <= int(question) <= len(dishes):
        dish_number = int(question) - 1 
        dish_name = dishes[dish_number].split('. ')[1]  
        detailed_question = f"Tell me how to make {dish_name}"
        response_dict = chain_recipes_instructions.invoke(detailed_question)
        clean_response = response_dict['result']
        print(f"Assistant: {clean_response}")
    else:
        response_dict = chain.invoke(question)
        clean_response = response_dict['result']
        print(f"Assistant: {clean_response}")
    print("******************************************************************\n")
