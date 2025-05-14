# Install required packages first (run these in terminal)
# pip install langchain-cohere
# pip install pydantic

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from pydantic import BaseModel
import os
from getpass import getpass

# 1. Define what information we want
class InstitutionDetails(BaseModel):
    founder: str
    founded: str
    branches: int
    employees: int
    summary: str

# 2. Set up the question template
prompt_template = """
Tell me about {institution_name} from Wikipedia:
1. Founder: 
2. Founded year: 
3. Number of branches: 
4. Number of employees: 
5. 4-line summary: 
"""

# 3. Get API key
if not os.environ.get("COHERE_API_KEY"):
    os.environ["COHERE_API_KEY"] = getpass("Enter Cohere API key: ")

# 4. Connect to Cohere AI
from langchain_cohere import ChatCohere
model = ChatCohere(model="command-r")

# 5. Create the question-answer system
prompt = PromptTemplate(input_variables=["institution_name"], template=prompt_template)
chain = LLMChain(llm=model, prompt=prompt)

# 6. Ask about an institution
institution_name = input("Enter institution name: ")
result = chain.invoke({"institution_name": institution_name})
print(result['text'])
