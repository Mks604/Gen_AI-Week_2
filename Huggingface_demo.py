#  🤖 Hugging Face Chatbot 

#!pip install transformers torch

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print("🤖 Hugging Face Chatbot (type 'exit' to stop)")

chat_history_ids = None

while True:
    user_input = input("You: ")
    
    if user_input.lower() == "exit":
        print("Bot: Goodbye!")
        break

    # Encode user input
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Append chat history
    bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids

    # Generate response
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode response
    bot_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    print("Bot:", bot_response)


#     🤖 Hugging Face Chatbot (type 'exit' to stop)
# You: Hello
# Bot: Hi! How are you today?
# You: What is AI?
# Bot: AI is artificial intelligence, which allows machines to learn and perform tasks.
# You: exit
# Bot: Goodbye!