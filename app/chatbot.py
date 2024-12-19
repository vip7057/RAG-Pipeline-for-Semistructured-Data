# from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings

from dotenv import load_dotenv
import os

load_dotenv()


os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
groq_api_key = os.getenv("GROQ_API_KEY")

st.title("ChatGroq Bot with llama3")

llm = ChatGroq(groq_api_key= groq_api_key, model_name = "Llama3-8b-8192")


## Prompt Template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only.
    Please provide the most accurate and relevant response to the question
    <context>
    {context}
    <context>
    Questions:{input}
    """
)


def vector_embeddings():

    if "vectors" not in st.session_state:

        st.session_state.embeddings = OllamaEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("./uni") #Ingest Data
        st.session_state.docs = st.session_state.loader.load() #Load Documents

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) #create chunks
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20]) #split documnets
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings) #vector ollama embeddings

        # Initialize retriever and store it in session state
        st.session_state.retriever = st.session_state.vectors.as_retriever(search_k=5)
        print(st.session_state.vectors)




prompt_question = st.text_input("Ask any question related to your Course Structure")


if st.button("Create Document Embedding"):
    vector_embeddings()
    st.write("FAISS Vector Store is now ready to be queried!")

if prompt_question:
    document_chain =  create_stuff_documents_chain(llm,prompt)
    retriever = st.session_state.as_retriever(search_k=5)
    retrieval_chain = create_retrieval_chain(retriever,document_chain)
    response = retrieval_chain.invoke({"input":prompt_question})

    st.write(response["answer"])


    with st.expander("Similarity Search within the Document"):
        #find relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("_________________________________________________________")



# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# # #
# # # ## Stremlit framework
# # # st.title("Your personal assistant for German University applications")
# # # input_txt = st.text_input("Ask me your Questions or Doubts")
# # #
# # #
# # # ## Ollama
# # # llm = Ollama(model="llama2")
# # # output_parser = StrOutputParser()
# # # chain = prompt|llm|output_parser #LangChain chain
# # #
# # # if input_txt:
# # #     st.write(chain.invoke({"question":input_txt}))
#
# # # Import necessary libraries
# # from langchain_openai import ChatOpenAI
# # from langchain_core.prompts import ChatPromptTemplate
# # from langchain_core.output_parsers import StrOutputParser
# # from langchain_community.llms import Ollama
# # import streamlit as st
# # from langchain_groq import ChatGroq
# # from langchain.text_splitter import RecursiveCharacterTextSplitter
# # from langchain.chains.combine_documents import create_stuff_documents_chain
# # from langchain.chains import create_retrieval_chain
# # from langchain_community.vectorstores import FAISS
# # from langchain_community.document_loaders import PyPDFDirectoryLoader
# # from langchain_community.embeddings import OllamaEmbeddings
# #
# # from dotenv import load_dotenv
# # import os
# # import faiss  # Import FAISS
# #
# # load_dotenv()
# #
# # # Load environment variables
# # os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
# # os.environ["LANGCHAIN_TRACING_V2"] = "true"
# # groq_api_key = os.getenv("GROQ_API_KEY")
# #
# # # Set up Streamlit app
# # st.title("ChatGroq Bot with Llama3")
# #
# # llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
# #
# # # Prompt Template
# # prompt = ChatPromptTemplate.from_template(
# #     """
# #     Answer the question based on the provided context only.
# #     Please provide the most accurate and relevant response to the question
# #     <context>
# #     {context}
# #     <context>
# #     Questions:{input}
# #     """
# # )
# #
# #
# # def load_faiss_vectors():
# #     """Load FAISS vectors from a file if they exist."""
# #     if os.path.exists("../data/faiss_index.bin"):
# #         st.session_state.vectors = faiss.read_index("faiss_index.bin")  # Load the FAISS index
# #         st.session_state.retriever = st.session_state.vectors.as_retriever(search_k=5)
# #         st.success("FAISS vectors loaded successfully!")
# #     else:
# #         st.warning("FAISS vectors not found. Please create them.")
# #
# #
# # def vector_embeddings():
# #     """Ingest documents and create FAISS vectors."""
# #     st.session_state.embeddings = OllamaEmbeddings()
# #     st.session_state.loader = PyPDFDirectoryLoader("./uni")  # Ingest Data
# #     st.session_state.docs = st.session_state.loader.load()  # Load Documents
# #
# #     st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)  # create chunks
# #     st.session_state.final_documents = st.session_state.text_splitter.split_documents(
# #         st.session_state.docs)  # split documents
# #     st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents,
# #                                                     st.session_state.embeddings)  # vector ollama embeddings
# #
# #     # Save the FAISS index to a file
# #     faiss.write_index(st.session_state.vectors.index, "faiss_index.bin")
# #
# #     # Initialize retriever and store it in session state
# #     st.session_state.retriever = st.session_state.vectors.as_retriever(search_k=5)
# #     st.success("FAISS Vector Store created and saved!")
# #
# #
# # # Load vectors if they exist; otherwise, prepare to create them
# # load_faiss_vectors()
# #
# # prompt_question = st.text_input("Ask any question related to your Course Structure")
# #
# # if st.button("Create Document Embedding"):
# #     vector_embeddings()
# #
# # if prompt_question and 'retriever' in st.session_state:
# #     document_chain = create_stuff_documents_chain(llm, prompt)
# #     retriever = st.session_state.retriever  # Access the initialized retriever
# #     retrieval_chain = create_retrieval_chain(retriever, document_chain)
# #     response = retrieval_chain.invoke({"input": prompt_question})
# #
# #     st.write(response["answer"])
# #
# #     with st.expander("Similarity Search within the Document"):
# #         # Find relevant chunks
# #         for i, doc in enumerate(response["context"]):
# #             st.write(doc.page_content)
# #             st.write("_________________________________________________________")
#
#
# # Import necessary libraries
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_community.llms import Ollama
# import streamlit as st
# from langchain_groq import ChatGroq
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# from bs4 import BeautifulSoup
# from langchain_community.embeddings import OllamaEmbeddings
#
# from dotenv import load_dotenv
# import os
# import faiss  # Import FAISS
#
# load_dotenv()
#
# # Load environment variables
# os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# groq_api_key = os.getenv("GROQ_API_KEY")
#
# # Set up Streamlit app
# st.title("ChatGroq Bot with Llama3")
#
# llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
#
# # Prompt Template
# prompt = ChatPromptTemplate.from_template(
#     """
#     Answer the question based on the provided context only.
#     Please provide the most accurate and relevant response to the question
#     <context>
#     {context}
#     <context>
#     Questions:{input}
#     """
# )
# #
# #
# # def load_faiss_vectors():
# #     """Load FAISS vectors from a file if they exist."""
# #     if os.path.exists("faiss_index.bin") and os.path.exists("faiss_store.pkl"):
# #         index = faiss.read_index("faiss_index.bin")  # Load the FAISS index
# #         st.session_state.vectors = FAISS(embedding_function=st.session_state.embeddings, index=index)
# #         st.session_state.vectors.load_local("faiss_store.pkl")  # Load documents from file
# #         st.session_state.retriever = st.session_state.vectors.as_retriever(search_k=5)
# #         st.success("FAISS vectors loaded successfully!")
# #     else:
# #         st.warning("FAISS vectors not found. Please create them.")
# #
# #
# # def vector_embeddings():
# #     """Ingest documents and create FAISS vectors."""
# #     st.session_state.embeddings = OllamaEmbeddings()
# #     st.session_state.loader = PyPDFDirectoryLoader("./uni")  # Ingest Data
# #     st.session_state.docs = st.session_state.loader.load()  # Load Documents
# #
# #     st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # create chunks
# #     st.session_state.final_documents = st.session_state.text_splitter.split_documents(
# #         st.session_state.docs[:50])  # split documents
# #     st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents,
# #                                                     st.session_state.embeddings)  # vector ollama embeddings
# #
# #     # Save the FAISS index and documents to file
# #     faiss.write_index(st.session_state.vectors.index, "faiss_index.bin")
# #     st.session_state.vectors.save_local("faiss_store.pkl")
# #
# #     # Initialize retriever and store it in session state
# #     st.session_state.retriever = st.session_state.vectors.as_retriever(search_k=5)
# #     st.success("FAISS Vector Store created and saved!")
# #
# #
# # # Load vectors if they exist; otherwise, prepare to create them
# # if "retriever" not in st.session_state:
# #     load_faiss_vectors()
# #
# # prompt_question = st.text_input("Ask any question related to your Course Structure")
# #
# # if st.button("Create Document Embedding"):
# #     vector_embeddings()
# #
# # if prompt_question and 'retriever' in st.session_state:
# #     document_chain = create_stuff_documents_chain(llm, prompt)
# #     retriever = st.session_state.retriever  # Access the initialized retriever
# #     retrieval_chain = create_retrieval_chain(retriever, document_chain)
# #     response = retrieval_chain.invoke({"input": prompt_question})
# #
# #     st.write(response["answer"])
# #
# #     with st.expander("Similarity Search within the Document"):
# #         # Find relevant chunks
# #         for i, doc in enumerate(response["context"]):
# #             st.write(doc.page_content)
# #             st.write("_________________________________________________________")
# # Setup Streamlit
#
#
# import requests
#
#
#
#
#
# # Embedding function setup
# def get_embeddings():
#     if "embeddings" not in st.session_state:
#         st.session_state.embeddings = OllamaEmbeddings()
#
#
# # Webpage scraping function
# def scrape_webpage(url):
#     """Scrape the content of a webpage using BeautifulSoup."""
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, "html.parser")
#
#     # Extract text content
#     paragraphs = soup.find_all('p')
#     content = " ".join([p.get_text() for p in paragraphs])
#
#     return content
#
#
# # Function to store and save FAISS vectors from webpage content
# def vector_embeddings_from_webpage(content, save_name="webpage_store.pkl"):
#     get_embeddings()
#
#     # Split the content into chunks
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     documents = [{"page_content": content}]
#     split_documents = text_splitter.split_documents(documents)
#
#     # Generate vectors
#     vectors = FAISS.from_documents(split_documents, st.session_state.embeddings)
#
#     # Save FAISS index and metadata
#     faiss.write_index(vectors.index, "webpage_faiss_index.bin")
#     vectors.save_local(save_name)
#
#     st.session_state.vectors = vectors
#     st.session_state.retriever = vectors.as_retriever(search_k=5)
#     st.success("Webpage content indexed and saved!")
#
#
# # Function to load FAISS vectors from local files
# def load_faiss_vectors(save_name="webpage_store.pkl"):
#     if os.path.exists("webpage_faiss_index.bin") and os.path.exists(save_name):
#         index = faiss.read_index("webpage_faiss_index.bin")  # Load FAISS index
#         st.session_state.vectors = FAISS(embedding_function=st.session_state.embeddings, index=index)
#         st.session_state.vectors.load_local(save_name)
#         st.session_state.retriever = st.session_state.vectors.as_retriever(search_k=5)
#         st.success("FAISS vectors loaded from saved data!")
#     else:
#         st.warning("FAISS vectors not found. Please scrape and create them.")
#
#
# # Input field for webpage URL
# webpage_url = st.text_input("Enter a webpage URL")
#
# # Button to scrape and create embeddings
# if st.button("Scrape and Create Embeddings"):
#     if webpage_url:
#         webpage_content = scrape_webpage(webpage_url)
#         vector_embeddings_from_webpage(webpage_content)
#     else:
#         st.error("Please enter a valid webpage URL.")
#
# # Button to load saved vectors
# if st.button("Load Saved Vectors"):
#     load_faiss_vectors()
#
# # Text input for querying the document
# query = st.text_input("Ask a question about the webpage content")
#
# # Query the FAISS retriever
# if query and 'retriever' in st.session_state:
#     document_chain = create_stuff_documents_chain(llm, prompt)
#     retriever = st.session_state.retriever  # Access the initialized retriever
#     retrieval_chain = create_retrieval_chain(retriever, document_chain)
#     response = retrieval_chain.invoke({"input": query})
#
#     st.write(response["answer"])
#
#     with st.expander("Similarity Search within the Document"):
#         # Find relevant chunks
#         for i, doc in enumerate(response["context"]):
#             st.write(doc.page_content)
#             st.write("_________________________________________________________")

import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt


# Function to fetch data from the API
def fetch_stock_data(symbol, api_key):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}"
    response = requests.get(url)
    data = response.json()

    # Check if the response contains 'Time Series (Daily)' key
    if "Time Series (Daily)" in data:
        time_series = data["Time Series (Daily)"]
        df = pd.DataFrame(time_series).T
        df['date'] = pd.to_datetime(df.index)
        df = df[['date', '4. close']].sort_values(by='date')
        df['4. close'] = pd.to_numeric(df['4. close'])
        return df
    else:
        st.error("Failed to retrieve data.")
        return None


# Streamlit App
def main():
    st.title("Stock Price Insights App")
    st.write("This app fetches stock prices and provides interactive insights.")

    # Sidebar input for stock symbol and API key
    st.sidebar.header("Settings")
    symbol = st.sidebar.text_input("Enter Stock Symbol (e.g. AAPL, MSFT)", "AAPL")
    api_key = st.sidebar.text_input("Enter API Key (get it from Alpha Vantage)", type="password")

    if api_key:
        st.write(f"Fetching data for {symbol}...")
        df = fetch_stock_data(symbol, api_key)

        if df is not None:
            st.write(f"Displaying data for {symbol}")

            # Show the first few rows of data
            st.write(df.head())

            # Plotting
            st.subheader(f"Stock Price History for {symbol}")
            plt.figure(figsize=(10, 5))
            plt.plot(df['date'], df['4. close'], label="Closing Price", color="blue")
            plt.xlabel("Date")
            plt.ylabel("Closing Price ($)")
            plt.title(f"{symbol} Stock Price Over Time")
            plt.xticks(rotation=45)
            plt.legend()
            st.pyplot(plt)

            # Statistical Insights
            st.subheader("Statistical Insights")
            st.write(f"**Mean Closing Price**: {df['4. close'].mean():.2f}")
            st.write(f"**Maximum Closing Price**: {df['4. close'].max():.2f}")
            st.write(f"**Minimum Closing Price**: {df['4. close'].min():.2f}")
            st.write(f"**Standard Deviation of Prices**: {df['4. close'].std():.2f}")

            # Sidebar slider for selecting date range
            st.sidebar.subheader("Select Date Range")
            min_date = df['date'].min().date()  # Convert to datetime.date
            max_date = df['date'].max().date()  # Convert to datetime.date
            start_date, end_date = st.sidebar.slider(
                "Adjust date range:",
                min_value=min_date,
                max_value=max_date,
                value=(min_date, max_date),
                format="YYYY-MM-DD"
            )

            # Filter data based on the date range from the sidebar slider
            filtered_data = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]

            st.write(filtered_data)

            # Plot the filtered data
            st.subheader(f"{symbol} Stock Price Between {start_date.date()} and {end_date.date()}")
            plt.figure(figsize=(10, 5))
            plt.plot(filtered_data['date'], filtered_data['4. close'], label="Closing Price", color="green")
            plt.xlabel("Date")
            plt.ylabel("Closing Price ($)")
            plt.title(f"{symbol} Stock Price Over Selected Date Range")
            plt.xticks(rotation=45)
            plt.legend()
            st.pyplot(plt)


# Run the app
if __name__ == "__main__":
    main()
