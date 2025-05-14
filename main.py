import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "Tianlin668/MentalT5"

## going to download and keep locally for faster access
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

## going to replace with an image reader to read text messages
input_text = input("please enter your message:")
inputs = tokenizer(input_text, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(inputs["input_ids"], max_length=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Response:", response)
