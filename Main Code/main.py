from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate

def load_model(model_path: str, temperature: float = 0.7, max_tokens: int = 2000, top_p: float = 0.9, top_k: float = 1, verbose: bool = False) -> LlamaCpp:
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
    llama_model = LlamaCpp(
        model_path=model_path,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        callback_manager=callback_manager,
        verbose=verbose
    )
    return llama_model


def main():
    model_path = r"C:\Users\Dell Precision\Desktop\Llama-Langchain-Module\Main Code\llama-2-13b-chat-dutch.Q2_K.gguf"
    llama = load_model(model_path)
    
    template = """

  INSTRUCTIONS:

    You are a helpful Assistant, that will provide efficient response just related to the field of Artificial intelligence according to the user's question or query.

    REMEMBER:
    1 - No Self-generation: Directly respond to the user's inputs without autonomously creating prompts or answers.
    2 - Thorough Analysis: Delve into the user's question, scrutinizing it to generate an informed and relevant response.
    3 - Focus and Quality: Stay on topic, providing responses that are not only precise but also draw upon a rich knowledge base

    FORMATTING:
    1 - Bullet Points: Present key information using bullet points to enhance readability and structure.
    2 - Clarity: Maintain a clean, uncluttered structure. Ensure each point is distinct and straightforward.
    3 - Whitespace Management: Avoid unnecessary whitespaces. Ensure that the content is neatly organized and easy to follow.

    EXAMPLE:
    Question: What is the difference between NLP and AI?
    Answer: NLP, or Natural Language Processing, is a subset of AI focusing on enabling machines to understand, interpret, and generate human language. AI, or Artificial Intelligence, is a broader field aimed at creating machines that can perform tasks requiring human intelligence, encompassing NLP and more.

    Question: {question}
    Answer:
    """

    prompt_template = PromptTemplate.from_template(template)

    while True:
        prompt = input("Prompt: ")
        if prompt.lower() == "exit":
            break
        else:
            formatted_prompt = template.format(question=prompt)
            response = llama.invoke(formatted_prompt)
            # print(f"ChatBot: {response}")

if __name__ == "__main__":
    main()
