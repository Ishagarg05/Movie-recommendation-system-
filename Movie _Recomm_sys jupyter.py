#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install sentence-transformers pandas scikit-learn matplotlib')


# In[2]:


import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np


# In[28]:


import pandas as pd

# Load uploaded file
df = pd.read_csv(r"C:\Users\DELL\Desktop\PUP_WEBSITE\FETCH\pup_all_results.csv")
# Check first few rows
df.head()


# In[29]:


df['combined'] = df[['department', 'exam_term', 'type']].astype(str).agg(' '.join, axis=1)


# In[30]:


df


# In[6]:


import re

def normalize_course(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'sem\s*[-]?[ivxlc]+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[\-\s]?\d{4}', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b20\d{2}\b', '', text)  # Remove years
    text = re.sub(r'\(.*?\)', '', text)      # Remove brackets
    text = re.sub(r'[^\w\s]', '', text)      # Remove punctuation
    return re.sub(r'\s+', ' ', text).strip()

# Normalize department names
df['normalized'] = df['department'].apply(normalize_course)
df.head()


# In[31]:


from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')


# In[32]:


from tqdm import tqdm
tqdm.pandas()  # For progress bar

# Convert to list
# department_texts = df['normalized'].tolist()
department_texts = df['combined'].tolist()
# Get embeddings
department_embeddings = model.encode(department_texts, show_progress_bar=True)


# In[35]:


department_embeddings


# In[36]:


query_embedding


# In[ ]:





# In[37]:


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Reshape user query to 2D for cosine_similarity
user_query_embedding = query_embedding.reshape(1, -1)

# Calculate cosine similarity
similarities = cosine_similarity(user_query_embedding, department_embeddings)

# Flatten to 1D array
similarities = similarities.flatten()

print("Cosine Similarities:", similarities)


# In[38]:


# Get top 3 most similar departments
top_3_indices = np.argsort(similarities)[::-1][:3]
print("Top 3 indices:", top_3_indices)
print("Top 3 similarities:", similarities[top_3_indices])


# In[39]:


from sklearn.metrics.pairwise import cosine_similarity

# Example user query
user_query = "btech cse"  # Change this as needed

# Normalize and encode query
# normalized_query = normalize_course(user_query)
query_embedding = model.encode(user_query)

# Cosine similarity
similarities = cosine_similarity([query_embedding], department_embeddings)[0]

# Add similarity to DataFrame
df['similarity'] = similarities

# Top 10 closest matches
top_matches = df.sort_values(by='similarity', ascending=False).head(10)
top_matches[['department', 'similarity']]


# In[40]:


from sklearn.metrics.pairwise import cosine_similarity

# Example user query
user_query = "btech cse sem 3 2022"  # Change this as needed

# Normalize and encode query
normalized_query = normalize_course(user_query)
query_embedding = model.encode([normalized_query])[0]

# Cosine similarity
similarities = cosine_similarity([query_embedding], department_embeddings)[0]

# Add similarity to DataFrame
df['similarity'] = similarities

# Top 10 closest matches
top_matches = df.sort_values(by='similarity', ascending=False).head(10)
top_matches[['department', 'similarity']]


# In[15]:


get_ipython().system('pip install pandas rapidfuzz')


# In[24]:


import pandas as pd
from rapidfuzz import fuzz, process

# Load CSV file
df = pd.read_csv(r"C:\Users\DELL\Desktop\PUP_WEBSITE\FETCH\pup_all_results.csv")

# Combine columns into a single string for better matching
df['combined'] = df[['department', 'exam_term', 'type']].astype(str).agg(' '.join, axis=1)






# In[27]:


df['combined'][0]


# In[23]:


# Example usage
user_input = "btech cse sem 3"
top_results = get_top_matches(user_input)

for res in top_results:
    print(res)


# In[ ]:




