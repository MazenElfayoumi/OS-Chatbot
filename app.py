import streamlit as st
import torch
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.llms import GPT4All
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate

# Check if a GPU is available and use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the local path for the GPT-4 model (ensure this is correctly configured for your environment)
local_path = "C:/Users/mezom/.cache/gpt4all/gpt4all-falcon-q4_0.gguf"

# Load documents and create the Faiss index
documents = TextLoader("D:/year_3/sem_1/Opretaing systems/project/data/mazen.txt", encoding='UTF-8').load_and_split()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
texts = text_splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
faiss_index = FAISS.from_documents(texts, embeddings)

# Set up the callback manager
callback_manager = BaseCallbackManager([StreamingStdOutCallbackHandler()])

# Load the GPT-4 model without device parameter
llm = GPT4All(model=local_path, callback_manager=callback_manager, verbose=True, repeat_last_n=0)

# Set up the prompt template
template = """
You are an artificial intelligence assistant.
The assistant gives helpful information about operating systems and only answers the questions that it knows in this domain. It doesn't answer personal questions.
Question: {question}\n\nAnswer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

# Create the Streamlit app
def main():
    st.title("OS chatbot")

    # Get user input
    question = st.text_input("Enter your question")

    # Generate the response
    if st.button("Get Answer"):
        with st.spinner("Generating Answer..."):
            response = llm_chain.run(question)
        st.success(response)

if __name__ == "__main__":
    main()