**Assessment Challenge: Retrieval-Augmented Generation (RAG) Q&A from a Technical Manual**

**Overview:**  
You are given a single technical manual (approximately 200 pages) in PDF format that contains both textual content and tabular data. Your task is to design and implement a Retrieval-Augmented Generation (RAG) pipeline that accurately answers user questions based on the manual’s content. Many queries will rely on information found in the manual’s tables, so careful consideration of how you handle and represent tabular data is essential.

We will provide a set of queries along with reference answers so you can test and refine your implementation.

### Constraints and Requirements

- **Open-Source LLM:** You must use an open-source Large Language Model. Proprietary or closed-source models are not allowed.
- **File Format:** Submit all code as `.py` files. Do not submit Jupyter notebooks.
- **User Interaction:** Provide a simple way for users (e.g., the evaluators) to interact with the RAG pipeline—this can be a command-line interface (CLI), a minimal REST API, or a simple Gradio interface. It does not need to be elaborate; just allow easy input of a query and display of the generated answer.
- **Evaluation Queries:** We will supply a set of queries and correct responses. Use these for your own testing and demonstration. Your solution does not need to perfectly match the reference answers, but should be reasonably close and show that you are effectively retrieving relevant context.

### Required Deliverables

1. **Architecture Design Document (1–2 pages)**  
   Describe your overall system design. Topics to cover:
   - **Document Ingestion & Representation:** How you plan to extract and represent both textual and tabular data from the PDF.  
   - **Chunking & Embedding Strategy:** How you will break the content into chunks, choose an embedding model, and store/query embeddings.  
   - **RAG Pipeline Flow:** How a user query leads to retrieval of relevant chunks and subsequent LLM-based answer generation. Include how you handle context limits and selection of relevant information.  
   - **Unanswerable Queries & Hallucinations:** Your approach for detecting when the required information is absent in the document and preventing misleading or hallucinatory answers.  
   - **Scalability, Performance & Maintenance:** Considerations for efficient querying, reduced complexity in the pipeline’s ongoing management, and approaches that facilitate easy maintenance or updates of components over time.
   - **Continuous Evaluation & Improvement:** How you might implement automated or systematic evaluation methods that can quickly reflect changes after model updates or modifications to the underlying documents, ensuring that quality and accuracy are tracked over time.

2. **Code Implementation**  
   Your code should:
   - **Content Extraction & Preparation:**  
     - Extract and prepare the PDF’s content for embedding.
   - **Embedding & Indexing:**  
     - Embed your data using an open-source embedding model.  
     - Build a vector-based retrieval index (e.g., using any suitable library or open-source vector database) to enable semantic search.  
   - **Retrieval & Generation Pipeline:**  
     - Given a user query, retrieve the most relevant document chunks.  
     - Construct a prompt for your chosen open-source LLM that incorporates the user’s query and the retrieved context.  
     - Generate a final answer and return it to the user.  
   - **User Interaction:**  
     - Implement a simple CLI, REST endpoint, or minimal Gradio interface to input queries and return answers.

3. **Evaluation & Demonstration**  
   - **Provided Queries & Reference Answers:**  
     We will give you a set of queries along with correct responses derived from the manual. Use these to test your pipeline’s accuracy.  
   - **Analysis & Discussion:**  
     After running your pipeline against the provided queries, briefly discuss how your results compare to the reference answers. Identify any limitations or error cases and suggest ways to improve accuracy and performance.

    **Note:**  
    The pipeline you build will likely not produce perfect answers for all the provided queries. The reference answers were obtained through a highly fine-tuned retrieval process, and it’s not expected that your solution will match them exactly. Getting all queries correct is not a requirement for this assessment. The queries and reference answers are provided primarily as a testing and diagnostic tool for you to gauge the effectiveness of your approach.

    What is important is that you understand the limitations of your solution. While you may not achieve perfect accuracy, demonstrating awareness of where your pipeline might fail, why it might fail, and proposing realistic improvements or mitigations will be highly valued in the evaluation process.

4. **Documentation**  
   - **README:**  
     - Instructions for setting up and running your code, including environment setup and dependencies.  
     - Summary of your design and methodological choices.  
     - Known limitations and areas for future improvement.

### Evaluation Criteria

- **Preprocessing & Data Handling:**  
  The thoroughness and correctness of PDF parsing, including how you handle and represent tables. The clarity and intelligence of your chunking strategy and how well it supports downstream retrieval.

- **Architecture Design:**  
  The clarity, logic, and completeness of your architectural design. How well your plan anticipates and addresses key challenges such as integrating table data, handling unanswerable queries, and managing prompt construction for the LLM.

- **Code Quality:**  
  Readability, maintainability, and clarity of your code. Efficient, well-structured, and well-documented code that demonstrates sound software engineering practices.

- **Evaluation & Reasoning:**  
  The quality of your analysis in comparing generated answers to reference answers. Thoughtful reflection on performance limitations and how they might be addressed.
