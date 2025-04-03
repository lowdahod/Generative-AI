!pip install sentence_transformers

!pip install langchain-huggingface

!pip install tf-keras --user

!pip install numpy==1.24.4 --user

from sentence_transformers import SentenceTransformer, util

# Load a pretrained SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define an expanded finance-related corpus
corpus = [
  "The stock market saw significant gains today, driven by strong earnings reports.",
  "Investing in diversified portfolios helps mitigate risk and maximize returns.",
  "The Federal Reserve's decision to raise interest rates could impact market liquidity.",
  "Cryptocurrency has become an increasingly popular asset class among investors.",
  "Financial analysts predict that the global economy will face challenges in the coming years.",
  "Bonds are considered a safer investment option compared to stocks.",
  "Banks are adopting blockchain technology to improve the efficiency of financial transactions.",
  "The economic impact of the pandemic has been severe, but recovery is underway.",
  "Inflation rates have been rising steadily, leading to higher costs for consumers.",
  "Corporate governance and transparency are crucial for investor confidence.",
  "The real estate market is experiencing a boom as demand outstrips supply in many areas."
]

# Encode the corpus into embeddings
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
corpus_embeddings


# Function to generate a story using contextual embeddings
def generate_paragraph(seed_word, corpus, corpus_embeddings, model,top_n=5):
  
  # Encode the seed word as a sentence
  seed_sentence = f"Tell me more about {seed_word} in finance."
  seed_embedding = model.encode(seed_sentence,convert_to_tensor=True)
  
  # Find the most similar sentences in the corpus to the seed sentence
  similarities = util.pytorch_cos_sim(seed_embedding,corpus_embeddings)[0]
  top_results = similarities.topk(top_n)
  print('top_results:',top_results)

  # Construct a more coherent story using the most similar sentences
  story = f"The topic of '{seed_word}' is crucial in the finance industry. "

  for idx in top_results.indices:
    similar_sentence = corpus[idx]
    story += f"{similar_sentence} "
  
  story += f"These concepts highlight the importance of {seed_word} in understanding financial markets and investment strategies."

  return story


# Example usage
seed_word = "Bonds"
story = generate_paragraph(seed_word, corpus, corpus_embeddings, model,
top_n=5)
print(story)
