from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from config.settings import MODEL_NAME

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Persistent memory for multi-turn conversation
chat_history = []

def generate_response(prompt):
    global chat_history

    # Append new user input to history
    chat_history.append(f"<|user|>\n{prompt}")
    
    # Construct full input from history
    full_prompt = "\n".join(chat_history) + "\n<|assistant|>\n"

    # Tokenize and generate response
    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=2048)
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )

    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Try to extract only assistant's latest message
    if "<|assistant|>" in decoded_output:
        response = decoded_output.split("<|assistant|>")[-1].strip()
    else:
        response = decoded_output.strip()

    # Save assistant reply to history
    chat_history.append(f"<|assistant|>\n{response}")
    return response
