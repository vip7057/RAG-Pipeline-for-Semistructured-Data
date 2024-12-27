# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# import torch
# from langchain_groq import ChatGroq
# from dotenv import load_dotenv
# import os
# import uuid
# from langchain_community.vectorstores import Chroma
# from langchain.storage import InMemoryStore
# from langchain.schema.document import Document
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.retrievers.multi_vector import MultiVectorRetriever
# from langchain_core.runnables import RunnablePassthrough, RunnableLambda
# from langchain_core.messages import SystemMessage, HumanMessage
#
# load_dotenv()
#
#
# os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# groq_api_key = os.getenv("GROQ_API_KEY")
# hf_token = os.getenv("HUGGINGFACE_TOKEN")
#
#
# from unstructured.partition.pdf import partition_pdf
#
# file_path = "./data/1706.03762v7.pdf"
#
# chunks = partition_pdf(
#     filename=file_path,
#     infer_table_structure=True,            # extract tables
#     strategy="hi_res",                     # mandatory to infer tables
#     chunking_strategy="by_title",          # or 'basic'
#     max_characters=4000,                  # defaults to 500
#     combine_text_under_n_chars=2000,       # defaults to 0
#     new_after_n_chars=3800,
#     extract_images_in_pdf=False,          # deprecated
#     # languages=["deu"],
# )
#
#
# # We get 2 types of elements from the partition_pdf function
# set([str(type(el)) for el in chunks])
#
# print(len(chunks))
#
#
# # # Each CompositeElement containes a bunch of related elements.
# # # This makes it easy to use these elements together in a RAG pipeline.
# from IPython.core.display import Markdown
#
# extracted_text = ""
#
# # print(chunks[2].metadata.orig_elements)
# chunk = chunks[1]
# for el in chunk.metadata.orig_elements:
#   if "Table" in str(type(el)):
#     extracted_text = extracted_text + str(el.metadata.text_as_html) + "</br></br>"
#   elif "Image" in str(type(el)):
#     extracted_text = extracted_text +  "</br></br>"
#   elif "Footer" in str(type(el)):
#     extracted_text = extracted_text +  "</br></br>"
#   else:
#     extracted_text = extracted_text + str(el) +  "</br></br>"
#
#
# print(extracted_text)
#
# tables = []
# texts = []
#
# for chunk in chunks:
#     # Initialize a temporary list to store text elements of the current chunk
#     temp_texts = []
#
#     for el in chunk.metadata.orig_elements:
#         if "Table" in str(type(el)):
#             # Append table element as HTML or text to the tables list
#             tables.append(el.metadata.text_as_html)
#         elif "Image" in str(type(el)) or "Footer" in str(type(el)):
#             # Skip Image and Footer elements
#             continue
#         else:
#             # Append all other elements as text to the temp_texts list
#             temp_texts.append(el.text)
#
#     # Combine all text elements of the current chunk into a single string
#     if temp_texts:
#         combined_text = "</br></br>".join(temp_texts)
#         texts.append(combined_text)
#
# # print(texts[5])
# # print(tables[3])
#
# # Prompt
# prompt_text = """
# You are an assistant tasked with summarizing tables and text.
# Give a concise summary of the table or text.
#
# Respond only with the summary, no additionnal comment.
# Do not start your message by saying "Here is a summary" or anything like that.
# Just give the summary as it is.
#
# Table or text chunk: {element}
#
# """
#
#
# # model = HuggingFacePipeline.from_model_id(
# #     model_id="mistralai/Mistral-7B-Instruct-v0.2",
# #     task="text-generation",
# #     device=-1,  # -1 for CPU
# #     batch_size=1,  # adjust as needed based on GPU map and model size.
# #     model_kwargs={"temperature": 0, "max_length": 4096, "torch_dtype":torch.bfloat16, "token":hf_token},
# # )
#
# model = ChatGroq(temperature=0.5, model="llama-3.1-8b-instant")
#
# print(model)
#
# prompt = ChatPromptTemplate.from_template(prompt_text)
#
# # Summary chain
# summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
#
# # Summarize text
# text_summaries = summarize_chain.batch(texts, {"max_concurrency": 2})
# # Summarize tables
# table_summaries = summarize_chain.batch(tables, {"max_concurrency": 2})
#
# # Use an open-source embedding model (e.g., Sentence Transformers)
# embedding_model = "sentence-transformers/all-MiniLM-L6-v2"  # You can choose other models as well
# embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
#
# # The vectorstore to use to index the child chunks
# vectorstore = Chroma(collection_name="multi_modal_rag", embedding_function=embeddings)
#
# # The storage layer for the parent documents
# store = InMemoryStore()
# id_key = "doc_id"
#
# # The retriever (empty to start)
# retriever = MultiVectorRetriever(
#     vectorstore=vectorstore,
#     docstore=store,
#     id_key=id_key,
# )
#
# # Add texts
# doc_ids = [str(uuid.uuid4()) for _ in texts]
# summary_texts = [
#     Document(page_content=summary, metadata={id_key: doc_ids[i]}) for i, summary in enumerate(text_summaries)
# ]
# retriever.vectorstore.add_documents(summary_texts)
# retriever.docstore.mset(list(zip(doc_ids, texts)))
#
# # Add tables
# table_ids = [str(uuid.uuid4()) for _ in tables]
# summary_tables = [
#     Document(page_content=summary, metadata={id_key: table_ids[i]}) for i, summary in enumerate(table_summaries)
# ]
# retriever.vectorstore.add_documents(summary_tables)
# retriever.docstore.mset(list(zip(table_ids, tables)))
#
# # Define the function to build the prompt
# def build_prompt(kwargs):
#     """Construct the prompt with text context."""
#     context_text = " ".join(kwargs["context"])  # Combine all text elements
#     user_question = kwargs["question"]
#
#     # Construct the prompt with the text context
#     prompt_template = f"""
#     Answer the question based only on the following context:
#     Context: {context_text}
#     Question: {user_question}
#     """
#     return ChatPromptTemplate.from_messages(
#         [HumanMessage(content=prompt_template)]
#     )
#
# # # Define the chain
# # model = ChatOpenAI(temperature=0, model_name="gpt-4")  # Initialize the model
#
# # chain = (
# #     {
# #         # Pass only the question to the retriever
# #         "context": RunnableLambda(lambda x: retriever.get_relevant_documents(x["question"])),
# #         "question": RunnablePassthrough(),  # Directly pass the user's question
# #     }
# #     | RunnableLambda(build_prompt)  # Build the prompt from context and question
# #     | model  # Pass the prompt to the model
# #     | StrOutputParser()
# # )
#
#
# chain = (
#     {
#         # Pass only the question to the retriever
#         "context": RunnableLambda(lambda x: retriever.get_relevant_documents(x["question"])),
#         "question": RunnablePassthrough(),  # Directly pass the user's question
#     }
#     | RunnableLambda(build_prompt)  # Build the prompt from context and question
#     | model  # Pass the prompt to the model
#     | StrOutputParser()
# )
#
# # chain =  {
# #         # Pass only the question to the retriever
# #         "context": RunnableLambda(lambda x: retriever.get_relevant_documents(x["question"])),
# #         "question": RunnablePassthrough(),  # Directly pass the user's question
# #     } | RunnablePassthrough().assign(
# #         response=(
# #     RunnableLambda(build_prompt)  # Build the prompt from context and question
# #     | model  # Pass the prompt to the model
# #     | StrOutputParser()
# #     )
# # )
#
# # Invoke the chain with the user question
# response = chain.invoke(
#     {
#         "question": "max path length for self-attention?"
#     }
# )
#
# print(response)


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
VECTORSTORE_PATH = "./vectorstore"
DOCSTORE_PATH = "./docstore"
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
        "question": "darf ich das Element H alleine verwenden?"
    }
)

print(response)
