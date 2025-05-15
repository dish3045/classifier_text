import torch
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def extract_messages_by_person(data, person_name):
    return [entry["text"] for entry in data if entry["sender"].lower() == person_name.lower()]

def detect_emotion(messages):
    combined_text = " ".join(messages)
    prompt = f"What is the emotional state of the person who said: \"{combined_text}\"?"

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(inputs["input_ids"], max_length=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

model_path = "./models"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

json_path = input("Enter path to the JSON file with messages: ")

with open(json_path, "r", encoding="utf-8") as f:
    chat_data = json.load(f)

senders = sorted(set(entry["sender"] for entry in chat_data))
print("\n👥 Available senders in the chat:")
for sender in senders:
    print("-", sender)

person = input("\nEnter the person's name to analyze: ")

person_msgs = extract_messages_by_person(chat_data, person)

if person_msgs:
    print(f"Messages from {person}:\n", person_msgs)
    emotion = detect_emotion(person_msgs)
    print(f"\nDetected Emotion for {person}: {emotion}")
else:
    print(f"\nNo messages found for '{person}' in the file.")