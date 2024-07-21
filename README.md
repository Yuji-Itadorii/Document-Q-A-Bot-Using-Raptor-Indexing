# Document Question Answering System using RAPTOR Indexing

## Objective

The objective of this project is to develop a system that extracts content from selected textbooks, creates a vector database using MILVUS with RAPTOR indexing, and develops a question-answering system using a Language Model (LLM).

## Project Structure

- `Data`: Folder that contains pdf file.
- `all-steps`: It is just a folder that contains the codes for all the steps separately.
- `helper.py`: It contains all the important functions for the app.py file.
- `qa_bot.py`: Streamlit application for the user interface.
- `app.py`: Script to create a Database from the pdf file.

## Setup

### Prerequisites

- Python 3.8 or higher
- `pip` package manager
- Hugging Face API Token

### Installation

1. Clone the repository:

    ```sh
    https://github.com/Yuji-Itadorii/Document-Q-A-Bot-Using-Raptor-Indexing.git
    cd your-repo-name
    ```

2. Install the required Python packages:

    ```sh
    pip install -r requirements.txt
    ```
3. Replace the 'YOUR_TOKEN' parameter with your API token in the required files

## Steps

### 1. Text Extraction

Extract text from PDF files using the PyPDF document loader from Langchain. Ensure that all relevant text is captured from the selected textbooks.

### 2. Data Chunking

Chunk the extracted text into smaller segments of approximately 512 tokens each, preserving sentence boundaries to maintain context.

### 3. Embedding and Indexing

Generate embeddings for the chunked data using Sentence-BERT. Apply RAPTOR indexing which includes clustering the embeddings using Gaussian Mixture Models (GMMs), summarizing the clusters using an LLM, and recursively creating a hierarchical tree structure.

### 4. Storing in MILVUS

Store the processed data in a MILVUS vector database. Implement Dense Passage Retrieval (DPR) to enhance the retrieval process from the database.

### 5. Question Answering System

Set up a question-answering system using an LLM. Import the stored data, use it to perform similarity searches, and generate relevant answers based on the retrieved documents.

### 6. User Interface

Develop a user interface using Streamlit. Allow users to input queries and display the retrieved answers along with the corresponding textbook title and page number.

## Running the Application

To run the application, execute the following command:

If you don't have a pre-created Milvus database locally then you must create the database first by running the following command otherwise skip this step:

```sh
python app.py
```
After the successful creation of the database the run following code:

```sh
streamlit run qa_bot.py
```

## Conclusion

This project demonstrates a comprehensive approach to extracting, processing, and utilizing textbook content for question-answering. By following the steps outlined above, you can replicate and extend this work to suit your needs.

Feel free to contribute, raise issues, or ask questions via GitHub.
