#!/usr/bin/env python3
import requests
import json

# Test root endpoint
print("Testing root endpoint...")
response = requests.get("http://localhost:8000/")
print(f"Root endpoint status: {response.status_code}")
print(f"Response: {response.json()}")

# Test query endpoint
print("\nTesting query endpoint...")
data = {
    'question': 'What is sustainable agriculture?',
    'device_id': 'test_device_123',
    'state': 'active',
    'language': 'en'
}

response = requests.post("http://localhost:8000/api/query", data=data)
print(f"Query endpoint status: {response.status_code}")
if response.status_code == 200:
    result = response.json()
    session = result.get('session', {})
    if session:
        messages = session.get('messages', [])
        if messages:
            print(f"Answer: {messages[0].get('answer', 'No answer found')}")
        print(f"Session ID: {session.get('session_id', 'No session ID')}")
        print(f"Language: {session.get('language', 'No language')}")
        print(f"State: {session.get('state', 'No state')}")
    else:
        print("No session data found")
else:
    print(f"Error: {response.text}")
