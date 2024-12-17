import os
import streamlit as st
from openai import AzureOpenAI

# Print environment info
print("Testing Azure OpenAI configuration...")
print("API Base:", st.secrets["AZURE_API_BASE"])
print("Deployment:", st.secrets["AZURE_DEPLOYMENT"])
print("API Version:", st.secrets["OPENAI_API_VERSION"])

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=st.secrets["AZURE_API_KEY"],
    api_version=st.secrets["OPENAI_API_VERSION"],
    azure_endpoint=st.secrets["AZURE_API_BASE"],
)

# Test completion
response = client.chat.completions.create(
    model=st.secrets["AZURE_DEPLOYMENT"],
    messages=[{"role": "user", "content": "Say hello!"}],
)

print("Response:", response.choices[0].message.content)
