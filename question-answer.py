import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

model = AutoModelForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
#print(model)

def get_answer(text, question):
    inputs = tokenizer(question, text, return_tensors='pt', max_length=512, truncation=True)
    outputs = model(**inputs)

    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))
    print(inputs['input_ids'])

    return answer

text = """
    The Eiffel Tower, located in Paris, France, is one of the most iconic landmarks in the world. It was designed by Gustave Eiffel and completed in 1889. The tower stands at a height of 324 meters and was the tallest man-made structure in the world at the time of its completion.
"""

question = "how high is the eiffel tower?"

answer = get_answer(text, question)
print(f"The answer to the question is: {answer}")
