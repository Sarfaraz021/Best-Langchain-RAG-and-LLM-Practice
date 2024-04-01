# LangChain supports many other chat models. Here, we're using Ollama
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# supports many more optional parameters. Hover on your `ChatOllama(...)`
# class to view the latest available supported parameters
llm = ChatOllama(model="mistral")
template = """
INSTRUCTIONS:

-You are an honest and helpful assistant. Your task is to provide quality responses to the user regarding any related query. 
-If you don't know anything about the question or query asked by the user, do not provide false information; instead, apologize to them.
-Be concise and to the point, addressing the user's specific question. 
-Do not repeat any question asked by the user, nor provide the same response multiple times. 
-Act like a professional chat assistant; if the user praises you, respond with, "It's my pleasure. Do you have any other questions? I   would be happy to assist."



{user_input}
Assistant: " "

"""
print("Chat Assistant \n")

while True:
    print("\n******************************************************************")
    user_input = input("User>: ")
    if user_input.lower() == "exit":
        print("Great to chat with you! By By.")
        break
    else:
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()
        response  = chain.invoke({"user_input": user_input})

        for chunks in chain.stream(response):
            print(chunks, end="", flush=True)

        print("Assitant>:" +response)
        print("******************************************************************\n")
        







# using LangChain Expressive Language chain syntax
# learn more about the LCEL on
# /docs/expression_language/why

# for brevity, response is printed in terminal
# You can use LangServe to deploy your application for
# production