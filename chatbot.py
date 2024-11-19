from transformers import AutoModelForCausalLM, AutoTokenizer, logging
import torch
import os

# Suppress irrelevant warnings
logging.set_verbosity_error()

# Set environment variable to avoid tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load model and tokenizer
model_name = "distilgpt2"  # Replace with a smaller model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize chat history with a system prompt
history = []
system_prompt = "You are a helpful and friendly assistant. Respond concisely and politely to all queries."
system_input_ids = tokenizer.encode(system_prompt + tokenizer.eos_token, return_tensors="pt")
history.append(system_input_ids)

# Chatbot loop
print("LLM Chatbot: Hi there! I'm your assistant. Type 'quit' to end the chat.")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        print("LLM Chatbot: Have a great day!")
        break

    # Tokenize input and add context
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    history.append(new_input_ids)

    # Concatenate the conversation history for context
    bot_input_ids = torch.cat(history, dim=-1)

    # Generate response
    response_ids = model.generate(
        bot_input_ids,
        max_new_tokens=100,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,  # Enable sampling for diverse output
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2
    )
    response = tokenizer.decode(response_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    print(f"LLM Chatbot: {response}")

    # Append response to history
    history.append(tokenizer.encode(response + tokenizer.eos_token, return_tensors="pt"))

