# this is so slow to run. maybe hours? 32gb swap memory usage. 
# I need to figure out how to make this work faster. 
#
# to run: 
# $ PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python chat-bot-v3.py
#

from transformers import AutoModelForCausalLM, AutoTokenizer

device = "mps" # the device to load the model onto MPS is the MBP "gpu"

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

messages = [
    {"role": "user", "content": "What is your favourite condiment?"},
    {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    {"role": "user", "content": "Do you have mayonnaise recipes?"}
]

encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

model_inputs = encodeds.to(device)
model.to(device)

generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])


