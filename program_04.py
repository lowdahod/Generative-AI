!pip install transformers -U
!pip install gensim
!pip install langchain_google_genai


!wget http://nlp.stanford.edu/data/glove.6B.zip
!unzip glove.6B.zip


!pip install numpy --upgrade --force-reinstall


!pip install --upgrade --force-reinstall numpy
!pip install --upgrade gensim
!pip install --upgrade scipy


import gensim.downloader as api

# Download and load pre-trained word2vec model
model = api.load('word2vec-google-news-300')

# Test the model
print(model.most_similar("king"))



# Define the original medical prompt
original_prompt = "Explain the importance of vaccinations in healthcare."

# Define key terms extracted from the original prompt
key_terms = ["vaccinations", "healthcare"]

# Initialize an empty list to store similar terms
similar_terms = []

# Loop through each key term to find similar words
for term in key_terms:
    try:
        # Get top 3 most similar words
        similar_words = model.most_similar(term, topn=3)
        similar_terms.extend([word for word, _ in similar_words])
    except KeyError:
        print(f"Term '{term}' not in vocabulary")

# Create enriched prompt
if similar_terms:
    enriched_prompt = f"{original_prompt} Consider aspects like: {', '.join(similar_terms)}"
else:
    enriched_prompt = original_prompt

print("Original Prompt:", original_prompt)
print("Enriched Prompt:", enriched_prompt)


import getpass
import os
from langchain_google_genai import ChatGoogleGenerativeAI

# Get API key
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

# Initialize the model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # Updated model name
    temperature=0,
    api_key=GOOGLE_API_KEY,
    max_tokens=256,
    timeout=None,
    max_retries=2
)

# Test the model
print(llm.invoke("Hi").content)



# Generate responses for both prompts
original_response = llm.invoke(original_prompt).content
enriched_response = llm.invoke(enriched_prompt).content

# Print comparison
print("=== ORIGINAL PROMPT RESPONSE ===")
print(original_response)
print("\n=== ENRICHED PROMPT RESPONSE ===")
print(enriched_response)


