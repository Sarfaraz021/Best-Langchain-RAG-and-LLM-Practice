import requests
import json

url = "http://localhost:11434/api/generate"

headers = {
    'Content-Type': 'application/json',
}

# Function to handle the interaction with the model
def generate_response(prompt):
    data = {
        "model": "llama2",
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    if response.status_code == 200:
        response_json = response.json()
        actual_response = response_json['response']
        print(actual_response)
        return actual_response
    else:
        print("Error:", response.text)
        return None

# Function to handle recipes
def handle_recipes():
    dishes_prompt = "List 10 popular dishes."
    print("Fetching dishes...")
    dishes = generate_response(dishes_prompt)
    return dishes.splitlines()

# Function to handle recipe details
def handle_recipe_details(dish_name):
    recipe_prompt = f"Provide the recipe instructions and ingredients for {dish_name}."
    print(f"Fetching recipe for {dish_name}...")
    recipe_details = generate_response(recipe_prompt)
    return recipe_details

# Main interaction loop
def main_loop():
    print("Welcome to the Recipe Chatbot!")
    while True:
        prompt = input("Enter 'recipes' for a dish list, a dish name for details, or anything else to chat:\n")

        if prompt.lower() in ['recipes', 'recipe']:
            dishes = handle_recipes()
            if dishes:
                print("\n".join(dishes))
            else:
                print("I couldn't get the dishes right now. Try again later.")
        elif prompt:
            # Check if the prompt is a dish from the last list
            recipe_details = handle_recipe_details(prompt)
            if recipe_details:
                print(recipe_details)
            else:
                print("I couldn't get the recipe details right now. Try again later.")
        else:
            # Default chat behavior
            chat_response = generate_response(prompt)
            if chat_response:
                print(chat_response)
            else:
                print("I couldn't process your request right now. Try again later.")
        
        # Option to continue or exit
        continue_chat = input("\nWould you like to continue? (yes/no):\n")
        if continue_chat.lower() != 'yes':
            break

# Remove the gradio interface launch and run the main loop instead
if __name__ == "__main__":
    main_loop()
