import requests

# Define the URL of your OLLAMA service
OLLAMA_URL = "http://localhost:1234/v1"

def query_ollama(prompt):
    # Define the payload with the prompt
    payload = {
        "prompt": prompt,
        "max_tokens": 50  # Adjust as needed
    }

    # Send a POST request to OLLAMA
    response = requests.post(OLLAMA_URL + "/completions", json=payload)
    print(response)
    # Check if the request was successful
    if response.status_code == 200:
        return response.json()["choices"][0]["text"]
    else:
        return None

# Example usage
prompt = "Once upon a time"
completion = query_ollama(prompt)
print(completion)
