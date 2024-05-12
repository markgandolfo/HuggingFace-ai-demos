import torch
from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis")

print("Sentiment Analysis with Huggingface Spaces")
print("Enter a sentance to analyse")

user_input = input()

if user_input:
    result = sentiment_pipeline(user_input)
    sentiment = result[0]['label']
    confidence = result[0]['score']

    print(f"Sentiment: {sentiment}")
    print(f"Confidence: {confidence}")
        
