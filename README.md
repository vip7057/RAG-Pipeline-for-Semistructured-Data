# RAG Pipeline for PDF chatbot

This project implements a Retrieval-Augmented Generation (RAG) pipeline for querying semi-structured PDFs conatining textual and tabular data.

## Setup and Installation

### Step 1: Clone the GitHub Repository
To start with the project, clone the GitHub repository to your local machine:

```bash
git clone https://github.com/vip7057/RAG-Pipeline-for-Semistructured-Data.git
```

Navigate to the project directory:

```bash
cd llm-chatbot-university-germany
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
Create a .env file in the project root directory. This file will store your API keys and tokens. 

Example of .env file:(A ".env" file with dummy keys and tokens is included in the directory, please add your langchain API key and HuggingFace token there)

``` bash
LANGCHAIN_API_KEY=your_langchain_api_key
HUGGINGFACE_TOKEN=your_huggingface_token
```

To obtain your API keys:

LangChain API Key:
Go to LangChain and sign up for an account. After signing in, find your API key in the dashboard and add it to the .env file.

HuggingFace Token:
Go to HuggingFace and sign up or log in. Navigate to your account settings, and under the Access Tokens tab, create a new token. Copy the token and add it to the .env file.


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
This will start the pipeline and allow you to interact with the application via a streamlit based web application.
(If you would like to add your own PDF files for querying then add them to "./app/data/", and then just enter the name of the PDF you want to query at the streamlit web app startup)

## System Design & Methodology
The overall methodology for this system is designed to process and answer user queries from PDFs efficiently.

#### PDF Partitioning & Chunking:
The first step is partitioning the PDF document using the unstructured library. Chunking by title is chosen because it logically separates different sections of the document based on headings, making it easier to organize and retrieve relevant content later. This approach allows the system to structure the data in a meaningful way, reducing irrelevant information in the query response.

#### Text and Table Separation:
- After partitioning, the text blocks are separated into two categories: raw text and tables. Tables are converted into HTML text format for better readability and extraction, as they often contain structured data that is crucial for queries related to tabular information.

#### Summarization of Raw Documents:
- The raw text and table content are then summarized using an open-source LLM. This step condenses the content while preserving key information, making it more manageable and suitable for retrieval. Summarization helps in reducing the size of the content, which is crucial for efficient querying.

#### Embedding & Storage:
- After summarization, the condensed versions of the text and table content are embedded into vectors using the Chroma Vectorstore. These embeddings are stored in the vector database, ensuring fast and efficient similarity search.
The corresponding raw documents (text and tables) are stored in an InMemory Docstore, enabling quick retrieval alongside their summarized versions.

#### Caching/Persisting for Reuse:
- Both the Vectorstore and Docstore are cached locally. This caching mechanism ensures that the same PDF doesn’t have to be processed multiple times, significantly improving system performance. Once a document has been processed, the embeddings and raw content are saved locally for future use, preventing redundant processing.

#### Query Processing & Retrieval:
- When a user submits a query, a "MultiVectorRetriever" is used to retrieve relevant summaries from the chroma vectorstore by indexing based on the query. This ensures that the most pertinent summaries are retrieved quickly.
The relevant raw documents are then retrieved from the docstore, providing detailed context to feed into the language model for generating an accurate response.

By organizing the workflow in this manner, the system efficiently processes large documents, retrieves relevant content, and generates answers to user queries with minimal latency, while the caching mechanism improves efficiency by avoiding redundant processing of previously analyzed PDFs.


## Limitations and Known Issues

### Table-Text Relevance:
The system stores tables separately from the surrounding text. This results in the loss of contextual relevance between the tables and the surrounding content. This is a known limitation and requires further improvements in how tables are embedded and related to the surrounding text.

### PDF Table Extraction:
The unstructured library works well for semi-structured documents but struggles with extracting tables from larger PDFs, especially those with complex layouts. As a result, the accuracy of table extraction may not always be optimal. Future work could involve integrating more advanced table extraction tools to improve the performance in these scenarios.

### Chunking and Querying:
The chunking strategy, although designed to manage large documents, may still face challenges when handling documents that require more granular chunking or specialized processing (e.g., scientific papers with many formulas).

### Summarization and Keyword Relevance:
While summarizing raw text and table documents for semantic retrieval, there is a risk that the summaries may not include the necessary keywords to match with the user's query. As a result, the retrieval step may completely miss out on relevant context. This can be mitigated by ensuring that the model is prompted to include all necessary keywords in the summaries to improve retrieval accuracy.


## Future Improvements

### Contextual Table Embedding:
Improve the system by embedding tables with the surrounding text to retain the relevance between text and tables. This will help improve the quality of responses, especially for queries related to specific tables.

### Enhanced Table Extraction:
Exploring advanced tools and libraries for better table extraction from large PDFs would enhance the system’s ability to process and handle complex documents more effectively.

### Query Processing Optimization:
Introducing a more robust query preprocessing pipeline, possibly with NLP-based techniques to better handle variations in user queries, could improve the response quality.

### Multilingual Support:
Adding support for multiple languages would expand the utility of the system for a broader audience.
