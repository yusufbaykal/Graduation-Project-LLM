# Turkish Natural Language Processing Solutions
This product leverages NLP (Natural Language Processing) and LLM (Large Language Models) technologies to streamline users' workflows and automate data preparation processes for developers.

It allows users to interact with their documents, engage in conversations with their desired websites, and eliminates time-consuming tasks such as report reading, article reading, and analysis extraction.

* The product includes 5 different modules.

# PDF ChatBot
A system that allows users to interact with their documents and reports, enabling them to retrieve specific information or generate summaries from within the document. It integrates an advanced RAG (Retrieval-Augmented Generation) system, ensuring users can efficiently access relevant content and insights without manually searching through the document.

# Web ChatBot
A system that generates outputs based on user prompts using the provided website as a data source. This solution also integrates an advanced RAG system, ensuring accurate and context-aware responses from the web content.

# Model Preparation
A system designed to streamline data preparation processes for developers. It applies built-in modules to the raw data, delivering cleaned and pre-processed output. Users can then download the ready-to-use data in CSV format, significantly reducing manual effort and preparation time.

# Text Statistics, Analysis & Preprocessing
This module automates tasks such as reading, interpreting, and processing user data. It provides an efficient solution for analyzing and preparing textual data, turning complex manual operations into seamless automated workflows.

# Environment Variables

* To run this project, you will need to add the following environment variables to your .env file:

OPENAI_API_KEY = ""

USER_AGENT = ["my_user_agent"]

# Data File

* Link: https://drive.google.com/drive/folders/1rp626oPDnZqOHurQAK0AJNpiBE2EgZq6?usp=sharing

  
## Run Locally

Clone the project:

```bash
  git clone https://github.com/yusufbaykal/Graduation-Project.git
```

## Usage

Navigate to the project directory:

```bash
  cd Graduation-Project
```
Install dependencies using Poetry:

```bash
  pip install poetry
  poetry lock 
  poetry install 

```
Navigate to the specific directory:

```bash
  cd Graduation-Project
  cd .\YazımDenetim\
```
Run the Streamlit application:

```python
  streamlit run streamlit_app.py
```

  

![Python](https://img.icons8.com/color/48/000000/python--v1.png)
![TensorFlow](https://img.icons8.com/color/48/000000/tensorflow.png)