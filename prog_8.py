 
!pip install langchain langchain-cohere langchain-community 


!pip install gdown 



import getpass
import os
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate

# Always ask for API key
api_key = getpass.getpass("Enter your Cohere API key: ")
os.environ["COHERE_API_KEY"] = api_key

# Initialize model
model = ChatCohere(model="command-r")  # or "command-r-plus"

# Create prompt template
prompt = ChatPromptTemplate.from_template("Tell me a quote about {topic}")
chain = prompt | model

# Invoke the chain
response = chain.invoke({"topic": "AI"})
print(response.content)



import gdown 
 
# Google Drive file ID (Extract from the URL) 
file_id = "18eab2igTTqdJLLnwmr8iDiCbYmA_aQ9C" 
file_path = "ai_agents_info.txt" 
 
# Download the file 
gdown.download(f"https://drive.google.com/uc?export=download&id={file_id}", file_path, quiet=False) 
 
# Read the file 
with open(file_path, "r", encoding="utf-8") as file: 
    document_text = file.read() 
 
print(document_text) 
