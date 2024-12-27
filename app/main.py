
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import uuid
from langchain_community.vectorstores import Chroma
# from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage
import pickle  # For saving and loading InMemoryStore
import json
import pdfplumber
import pandas as pd
from unstructured.partition.pdf import partition_pdf
import streamlit as st

# Load environment variables
load_dotenv()
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
hf_token = os.getenv("HUGGINGFACE_TOKEN")

# Constants for file paths and PDF name-based stores
BASE_PATH = "./data"
VECTORSTORE_BASE_PATH = "./vectorstores"
DOCSTORE_BASE_PATH = "./docstores"
PDF_FILE_PATH = os.path.join(BASE_PATH, "1706.03762v7.pdf")

# Define the embedding model
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)


# Function to initialize the LLM model
def init_model():
    """Initialize and return the HuggingFace LLM model."""
    model = HuggingFacePipeline.from_model_id(
        model_id="mistralai/Mistral-7B-Instruct-v0.2",
        task="text-generation",
        device=-1,  # -1 for CPU
        batch_size=1,  # adjust as needed based on GPU map and model size.
        model_kwargs={"temperature": 0, "max_length": 4096, "torch_dtype":torch.bfloat16, "token":hf_token},
    )
    return model


# Function to process the PDF into chunks
def process_pdf(file_path):
    """Extract chunks of content from the provided PDF."""
    return partition_pdf(
        filename=file_path,
        infer_table_structure=True,
        strategy="hi_res",
        chunking_strategy="by_title",
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
        extract_images_in_pdf=False,
    )


# Function to extract and summarize the text and tables from chunks
def extract_and_summarize(chunks):
    """Summarize text and table chunks."""
    tables, texts = [], []

    for chunk in chunks:
        temp_texts = []
        for el in chunk.metadata.orig_elements:
            if "Table" in str(type(el)):
                tables.append(el.metadata.text_as_html)
            elif "Image" in str(type(el)) or "Footer" in str(type(el)):
                continue
            else:
                temp_texts.append(el.text)
        if temp_texts:
            combined_text = "</br></br>".join(temp_texts)
            texts.append(combined_text)

    # Summarize text and table chunks
    summarize_chain = create_summarize_chain()
    text_summaries = summarize_chain.batch(texts, {"max_concurrency": 1})
    table_summaries = summarize_chain.batch(tables, {"max_concurrency": 1})

    return text_summaries, table_summaries, texts, tables


# Function to create a summarization chain
def create_summarize_chain():
    """Create a summarization chain using a predefined prompt and model."""
    prompt_text = """
    You are an assistant tasked with summarizing tables and text that are completely in German.
    Give a concise summary of the table or text EXPLICITLY in German.

    Respond only with the summary, no additionnal comment.
    Do not start your message by saying "Here is a summary" or anything like that.
    Just give the summary as it is.

    Table or text chunk: {element}

    """
    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = init_model()
    return {"element": lambda x: x} | prompt | model | StrOutputParser()


# Function to generate the filename-based docstore and vectorstore paths
def generate_store_paths(pdf_name):
    """Generate paths for the docstore and vectorstore based on the PDF name."""
    base_name = pdf_name.split('.')[0]  # remove file extension
    vectorstore_path = os.path.join(VECTORSTORE_BASE_PATH, f"{base_name}_vectorstore")
    docstore_path = os.path.join(DOCSTORE_BASE_PATH, f"{base_name}_docstore")
    return vectorstore_path, docstore_path


# Function to chunk, embed, and store summaries in vectorstore
def chunk_and_embed(text_summaries, table_summaries, texts, tables, pdf_name):
    """Embed and store the summaries in vectorstore and docstore."""
    vectorstore_path, docstore_path = generate_store_paths(pdf_name)

    # Ensure that the paths are created
    # os.makedirs(vectorstore_path, exist_ok=True)
    os.makedirs(docstore_path, exist_ok=True)

    # Initialize the vectorstore and store
    vectorstore = Chroma(collection_name="multi_modal_rag", embedding_function=embeddings,
                         persist_directory=vectorstore_path)
    store = InMemoryStore()

    # Document IDs
    doc_ids = [str(uuid.uuid4()) for _ in texts]
    summary_texts = [Document(page_content=summary, metadata={"doc_id": doc_ids[i]}) for i, summary in
                     enumerate(text_summaries)]
    vectorstore.add_documents(summary_texts)
    store.mset(list(zip(doc_ids, texts)))

    # Process and store table summaries
    table_ids = [str(uuid.uuid4()) for _ in tables]
    summary_tables = [Document(page_content=summary, metadata={"doc_id": table_ids[i]}) for i, summary in
                      enumerate(table_summaries)]
    vectorstore.add_documents(summary_tables)
    store.mset(list(zip(table_ids, tables)))

    # Save data for future use
    save_data(doc_ids + table_ids, texts + tables, docstore_path)



# Function to save document IDs and content
def save_data(doc_ids, docs, store_path):
    """Save the document IDs and content to disk as json for reuse."""
    with open(os.path.join(store_path, "doc_ids.json"), "w") as f:
        json.dump(doc_ids, f)

    with open(os.path.join(store_path, "docs.json"), "w") as f:
        json.dump(docs, f)


# Function to load stored data
def load_data(pdf_name):
    """Load the vectorstore and docstore based on the PDF name."""
    vectorstore_path, docstore_path = generate_store_paths(pdf_name)

    if os.path.exists(vectorstore_path) and os.path.exists(docstore_path):
        loaded_vectorstore = Chroma(collection_name="multi_modal_rag", persist_directory=vectorstore_path,
                                    embedding_function=embeddings)
        loaded_store = InMemoryStore()

        with open(os.path.join(docstore_path, "doc_ids.json"), "r") as f:
            doc_ids = json.load(f)

        with open(os.path.join(docstore_path, "docs.json"), "r") as f:
            docs = json.load(f)

        loaded_store.mset(list(zip(doc_ids, docs)))
        retriever = MultiVectorRetriever(vectorstore=loaded_vectorstore, docstore=loaded_store, id_key="doc_id")

        return retriever
    return None


# Function to build a prompt for the user query
def build_prompt(kwargs):
    """Build a prompt using the retrieved context and the user query."""
    context_text = " ".join(kwargs["context"])
    user_question = kwargs["question"]
    # prompt_template = f"""
    # Answer the question based only on the following context:
    # Context: {context_text}
    # Question: {user_question}
    # """
    prompt_template = f"""
           Answer in German to the question in German based only on the following context, DO NOT ANSWER IF YOU ARE NOT SURE OR YOU DONT HAVE A RELEVANT CONTEXT:
           Context: {context_text}
           Question: {user_question}
           """
    return ChatPromptTemplate.from_messages(
        [HumanMessage(content=prompt_template)]
    )


# Function to process the query and generate a response
def process_query(query, retriever, model):
    """Process a user query using the retriever and LLM model."""
    chain = (
        {
            "context": RunnableLambda(lambda x: retriever.get_relevant_documents(x["question"])),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(build_prompt)
        | model
        | StrOutputParser()
    )

    # chain = {
    #             # Pass only the question to the retriever
    #             "context": RunnableLambda(lambda x: retriever.get_relevant_documents(x["question"])),
    #             "question": RunnablePassthrough(),  # Directly pass the user's question
    #         } | RunnablePassthrough().assign(
    #     response=(
    #             RunnableLambda(build_prompt)  # Build the prompt from context and question
    #             | model  # Pass the prompt to the model
    #             | StrOutputParser()
    #     )
    # )

    response = chain.invoke(
        {
            "question": str(query)
        }
    )
    return response

def process_and_load(pdf_name):
    """
    Process the PDF and create stores if they don't exist, then load the retriever.
    """
    # Replace this path with the actual PDF file location
    pdf_path = os.path.join(BASE_PATH, pdf_name)

    if not os.path.exists(pdf_path):
        st.error(f"PDF file '{pdf_name}' not found. Please ensure the file exists in the 'data' directory.")
        return None

    # Process the PDF and save data
    chunks = process_pdf(pdf_path)
    text_summaries, table_summaries, texts, tables = extract_and_summarize(chunks)
    chunk_and_embed(text_summaries, table_summaries, texts, tables, pdf_name)
    st.success(f"PDF '{pdf_name}' processed and data stored successfully.")
    return load_data(pdf_name)


# Updated Streamlit Interface with Auto-Processing
def main():
    st.title("RAG Q&A with Technical Manual")

    # Input to specify or select the PDF file name
    st.subheader("Enter or select the PDF file to process:")
    pdf_name = st.text_input("PDF Name (e.g., 'technical_manual.pdf'):")

    if not pdf_name:
        st.warning("Please enter the name of the PDF file.")
        return

    # Generate the store paths based on the entered PDF name
    vectorstore_path, docstore_path = generate_store_paths(pdf_name)

    # Check if vectorstore and docstore already exist
    if os.path.exists(vectorstore_path) and os.path.exists(docstore_path):
        st.success(f"Data for '{pdf_name}' found. Loading existing stores.")
        retriever = load_data(pdf_name)
    else:
        st.info(f"No existing data found for '{pdf_name}'. Processing the PDF automatically.")
        retriever = process_and_load(pdf_name)

    # Load the model
    model = init_model()

    if retriever:
        query = st.text_input("Enter your question:")
        if query:
            response = process_query(query, retriever, model)
            st.write(f"Answer: {response}")


if __name__ == "__main__":
    main()
