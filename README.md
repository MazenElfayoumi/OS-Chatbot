# OS Chatbot

## Overview

The OS Chatbot is a web application developed using Streamlit and LangChain to provide answers to questions related to operating systems. The chatbot leverages a local GPT-4 model and various NLP techniques to offer helpful information. The application is designed to be interactive and user-friendly, running in a web browser.

## Technologies Used

- **Backend:** Python
- **Machine Learning:** GPT-4
- **Text Processing:** LangChain
- **Web Interface:** Streamlit
- **Vector Store:** FAISS
- **Embeddings:** HuggingFace

## Features

- **Interactive UI:** Built with Streamlit for a seamless user experience.
- **Local GPT-4 Model:** Provides responses to questions related to operating systems.
- **Real-time Interaction:** Efficient handling of user queries and responses.
- **Document Processing:** Uses FAISS and embeddings for text retrieval.

## Project Structure

- `app.py`: Main Streamlit application file.
- `data/`: Directory containing data files for processing.
- `requirements.txt`: Lists the Python packages required for the project.

## Showcases

Below are two showcases demonstrating the chatbot in action:

### Showcase 1

![result1](https://github.com/user-attachments/assets/bfbb36b9-cd0c-4c54-a88e-9d20128ffbe1)


### Showcase 2

![result2](https://github.com/user-attachments/assets/1d8c1692-91ae-4413-9340-4f43b5e19217)


*Replace `path/to/showcase1.png` and `path/to/showcase2.png` with the actual paths to your image files.*

## Usage

1. **Run the Streamlit App**

   ```bash
   streamlit run app.py
2. **Interact with the Chatbot**

Open your web browser and navigate to `http://localhost:8501` to interact with the chatbot.
