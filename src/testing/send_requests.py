import requests
import json

url = "http://127.0.0.1:8000/ask"
while (1):
    query: str =  input("Enter your query, or quit to quit:\n")
    if query == "quit": break
    payload = {
        "question": query
    }
    headers = {
        "Content-Type": "application/json"
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        print("Request successful!")
        print("Response:\n", response.text)
    else:
        print(f"Request failed with status code: {response.status_code}")
        print("Response text:", response.text)