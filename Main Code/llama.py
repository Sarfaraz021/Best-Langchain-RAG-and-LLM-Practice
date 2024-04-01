# from langchain.callbacks.manager import CallbackManager
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# from langchain_community.llms import LlamaCpp
# from langchain_core.prompts import PromptTemplate
# from langchain.chains import ConversationChain
# from langchain.memory import ConversationBufferMemory

# def load_model(model_path: str, temperature: float = 0.7, max_tokens: int = 2000, top_p: float = 0.9, top_k: float = 1, verbose: bool = False) -> LlamaCpp:
#     callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
#     llama_model = LlamaCpp(
#         model_path=model_path,
#         temperature=temperature,
#         max_tokens=max_tokens,
#         top_p=top_p,
#         callback_manager=callback_manager,
#         verbose=verbose
#     )
#     return llama_model

# def main():
#     model_path = r"C:\Users\Dell Precision\Desktop\Llama-Langchain-Module\Main Code\llama-2-13b-chat-dutch.Q2_K.gguf"
#     llama = load_model(model_path)

#     template = """
#     INSTRUCTIONS:

#     You are a helpful Assistant, that will provide efficient response just related to the field of Artificial intelligence according to the user's question or query.

#     REMEMBER:
#     1 - Do not generate prompts and answers yourself.
#     2 - Brainstorm the user's given query or question, analyze it, and then generate a response accordingly.
#     3 - Stick to the point and provide quality responses with the advanced data you have.

#     EXAMPLE:
#     Question: What is the difference between NLP and AI?
#     Answer: NLP, or Natural Language Processing, is a subset of AI focusing on enabling machines to understand, interpret, and generate human language. AI, or Artificial Intelligence, is a broader field aimed at creating machines that can perform tasks requiring human intelligence, encompassing NLP and more.

#     {chat_history}"""

#     prompt_template = PromptTemplate.from_template(template)
#     memory = ConversationBufferMemory(memory_key="chat_history")
#     conversation = ConversationChain(llm=llama, prompt=prompt_template, memory=memory)

#     while True:
#         prompt = input("Prompt: ")
#         if prompt.lower() == "exit":
#             break
#         else:
#             response = conversation.run(input=prompt)
#             print(f"ChatBot: {response}")

# if __name__ == "__main__":
#     main()
