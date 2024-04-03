import requests
import json

url = "http://localhost:11434/api/generate"

headers = {
    'Content-Type': 'application/json',
}

def generate_response(prompt):
    data = {
        "model": "llama2", 
        # "model": "mistral", 
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        data = response.json()
        actual_response = data['response']
        return actual_response
    else:
        # Print the error from the API
        print("Error:", response.text)
        return None

# The main interaction loop
while True:
    user_input = input("Enter your prompt: ")
    if user_input.lower() == 'exit':
        print("Exiting the program.")
        break
    response = generate_response(user_input)
    if response:
        print(response)
