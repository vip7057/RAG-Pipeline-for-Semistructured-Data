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





load_dotenv()

os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
groq_api_key = os.getenv("GROQ_API_KEY")
hf_token = os.getenv("HUGGINGFACE_TOKEN")

model = ChatGroq(temperature=0.5, model="llama-3.1-8b-instant") #llama-3.1-8b-instant

# Paths for saving/loading data
VECTORSTORE_PATH = "./vectorstores/technical_manual_vectorstore"
DOCSTORE_PATH = "./docstores/technical_manual_docstore"
DOCIDS_FILE = os.path.join(DOCSTORE_PATH, "doc_ids.json")
DOCS_FILE = os.path.join(DOCSTORE_PATH, "docs.json")

# Set up embeddings
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

if os.path.exists(VECTORSTORE_PATH) and os.path.exists(DOCIDS_FILE) and os.path.exists(DOCS_FILE):
    # Reload the vectorstore
    loaded_vectorstore = Chroma(collection_name="multi_modal_rag", persist_directory=VECTORSTORE_PATH, embedding_function=embeddings)


    # Reload the InMemoryStore docstore
    with open(DOCIDS_FILE, "r") as f:
        doc_ids = json.load(f)
        print(doc_ids)

    with open(DOCS_FILE, "r") as f:
        docs = json.load(f)
        print(docs)

    loaded_store = InMemoryStore()
    loaded_store.mset(list(zip(doc_ids, docs)))

    # Recreate the retriever
    retriever = MultiVectorRetriever(
        vectorstore=loaded_vectorstore,
        docstore=loaded_store,
        id_key="doc_id",
    )

    print("Data loaded successfully.")

else:
    # Process PDF and create embeddings if data is not saved

    file_path = "./data/technical_manual.pdf"

    chunks = partition_pdf(
        filename=file_path,
        infer_table_structure=True,
        strategy="hi_res",
        chunking_strategy="by_title",
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
        extract_images_in_pdf=False,
    )

    tables = []
    texts = []
    tables_pp = []

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

    # with pdfplumber.open(file_path) as pdf:
    #     for page in pdf.pages:
    #         # Extract tables from each page
    #         table = page.extract_tables()
    #         if table:
    #             # Iterate over the tables extracted from the page
    #             for t in table:
    #                 # Convert table to DataFrame
    #                 df = pd.DataFrame(t[1:], columns=t[0])  # Use the first row as column names
    #                 # Convert DataFrame to HTML
    #                 html_table = df.to_html(index=False)
    #                 # Append the HTML version of the table to the tables list
    #                 tables_pp.append(html_table)

    # prompt_text = """
    # You are an assistant tasked with summarizing tables and text.
    # Give a concise summary of the table or text.
    #
    # Respond only with the summary, no additional comment.
    # Do not start your message by saying "Here is a summary" or anything like that.
    # Just give the summary as it is.
    #
    # Table or text chunk: {element}
    # """

    # Prompt
    prompt_text = """
    You are an assistant tasked with summarizing tables and text that are completely in German.
    Give a concise summary of the table or text EXPLICITLY in German.

    Respond only with the summary, no additionnal comment.
    Do not start your message by saying "Here is a summary" or anything like that.
    Just give the summary as it is.

    Table or text chunk: {element}

    """
    prompt = ChatPromptTemplate.from_template(prompt_text)
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    text_summaries = summarize_chain.batch(texts, {"max_concurrency": 1})
    table_summaries = summarize_chain.batch(tables, {"max_concurrency": 1})

    vectorstore = Chroma(collection_name="multi_modal_rag", embedding_function=embeddings, persist_directory=VECTORSTORE_PATH)
    store = InMemoryStore()

    doc_ids = [str(uuid.uuid4()) for _ in texts]
    summary_texts = [
        Document(page_content=summary, metadata={"doc_id": doc_ids[i]}) for i, summary in enumerate(text_summaries)
    ]
    vectorstore.add_documents(summary_texts)
    store.mset(list(zip(doc_ids, texts)))

    table_ids = [str(uuid.uuid4()) for _ in tables]
    summary_tables = [
        Document(page_content=summary, metadata={"doc_id": table_ids[i]}) for i, summary in enumerate(table_summaries)
    ]
    vectorstore.add_documents(summary_tables)
    store.mset(list(zip(table_ids, tables)))

    # Save `doc_ids` and `docs` for later use
    os.makedirs(DOCSTORE_PATH, exist_ok=True)

    with open(DOCIDS_FILE, "w") as f:
        json.dump(doc_ids + table_ids, f)

    with open(DOCS_FILE, "w") as f:
        json.dump(texts + tables, f)

    # vectorstore.persist()
    print("Data persisted successfully.")

    # # Reload the InMemoryStore docstore
    # with open(DOCIDS_FILE, "r") as f:
    #     doc_ids = json.load(f)
    #
    # with open(DOCS_FILE, "r") as f:
    #     docs = json.load(f)


    retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key="doc_id")

def build_prompt(kwargs):
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



# chain = (
#     {
#         "context": RunnableLambda(lambda x: retriever.invoke(x["question"])),
#         "question": RunnablePassthrough(),
#     }
#     | RunnableLambda(build_prompt)
#     | model
#     | StrOutputParser()
# )

chain = {
        # Pass only the question to the retriever
        "context": RunnableLambda(lambda x: retriever.get_relevant_documents(x["question"])),
        "question": RunnablePassthrough(),  # Directly pass the user's question
    } | RunnablePassthrough().assign(
        response=(
    RunnableLambda(build_prompt)  # Build the prompt from context and question
    | model  # Pass the prompt to the model
    | StrOutputParser()
    )
)

response = chain.invoke(
    {
        "question": "gibt es die Isokorb K-O-M4-V3?"

    }
)

print(response)
