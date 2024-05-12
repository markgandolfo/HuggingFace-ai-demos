from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
model_name = "facebook/blenderbot-400M-distill"

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

conversation_history = ["a chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions." ]

while True:
    history_string = "\n".join(conversation_history)

    input_text = input("> ")

    inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")
    tokenizer.pretrained_vocab_files_map

    output = model.generate(**inputs)

    response = tokenizer.decode(output[0], skip_special_tokens=True).strip()

    print(f"assistant: {response}")

    conversation_history.append(f"user: {input_text}")
    conversation_history.append(f"assistant: {response}")

