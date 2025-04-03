!pip install --upgrade --quiet huggingface_hub
!pip install --upgrade langchain
!pip install transformers
!pip install langchain-huggingface



from transformers import pipeline

# Load the sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

# Example sentences for analysis
sentences = [
    "The product quality is amazing! I'm very satisfied.",
    "I had a terrible experience with customer service.",
    "The delivery was quick, but the packaging was damaged.",
    "Absolutely love this! Best purchase I've made.",
    "Not worth the money, very disappointed."
]

# Analyze sentiment for each sentence
results = sentiment_analyzer(sentences)

# Print the results
for sentence, result in zip(sentences, results):
    print(f"Sentence: {sentence}\nSentiment: {result['label']}, Confidence: {result['score']:.2f}\n")


