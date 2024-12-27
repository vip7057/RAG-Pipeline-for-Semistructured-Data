# RAG Pipeline for PDF chatbot

This project implements a Retrieval-Augmented Generation (RAG) pipeline for querying semi-structured PDFs conatining textual and tabular data.

## Setup and Installation

### Step 1: Clone the GitHub Repository
To start with the project, clone the GitHub repository to your local machine:

```bash
git clone https://github.com/vip7057/llm-chatbot-university-germany.git
```

Navigate to the project directory:

```bash
cd repo-name
```

### Step 2: Set up a Conda Environment
First, create a Python environment (version 3.9.20 or higher) for the project. This ensures compatibility with the required libraries.


1. Create a new conda environment:

```bash
conda create -n rag-pipeline python=3.9.20
```

2. Activate the environment:
```bash
conda activate rag-pipeline
```


### Step 3: Install Required Dependencies
Once the environment is set up and activated, install the dependencies listed in the requirements.txt file:

```bash
pip install -r requirements.txt
```

### Step 4: Set Up the .env File
Create a .env file in the project root directory. This file will store your API keys and tokens. It should contain the following variables:

![LANGCHAIN_API_KEY: Your LangChain API key.]
![HUGGINGFACE_TOKEN: Your HuggingFace token for using models like bigscience/bloom-560m.]

To obtain your API keys:

LangChain API Key:
Go to LangChain and sign up for an account. After signing in, find your API key in the dashboard and add it to the .env file.

HuggingFace Token:
Go to HuggingFace and sign up or log in. Navigate to your account settings, and under the Access Tokens tab, create a new token. Copy the token and add it to the .env file.

Example of .env file:

``` bash
LANGCHAIN_API_KEY=your_langchain_api_key
HUGGINGFACE_TOKEN=your_huggingface_token
```
### Step 5: Install Additional Tools
The following external tools are required for extracting data from PDFs and performing OCR tasks:

#### Tesseract OCR Installation

- Windows: 
Download the installer from the web. Run the installer and add the Tesseract binary to your system's PATH.

- Linux: 
Install Tesseract using the following command:

```bash
sudo apt install tesseract-ocr
```
- Mac:
Install Tesseract via Homebrew:

```bash
brew install tesseract
```

#### Poppler Installation (required for PDF parsing)

- Windows:
Download the Poppler binaries from the web. Add the bin directory of Poppler to your system’s PATH.

- Linux:
Install Poppler using the following command:

```bash
sudo apt install poppler-utils
```

- Mac:
Install Poppler via Homebrew:
```bash
brew install poppler
```
### Step 6: Run the Code
After setting up the environment and installing dependencies, you can run the code by navigating to the appropriate directory and starting the main script.

Navigate to the app directory:
```bash
cd app
```

Run the main.py file:
```bash
streamlit run main.py
```
This will start the pipeline and allow you to interact with the application via a web application.
