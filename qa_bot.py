from langchain_community.vectorstores import Milvus
from langchain_community.llms import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer
import streamlit as st

st.title("Q&A Bot with DPR Embeddings and Milvus Vector Store")
st.write("Ask Question Realted to Document")

# Load DPR models and tokenizers
question_encoder = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
context_encoder = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

# Create DPR embeddings class DPR (Dense Passage Retrieval)
class DPRHuggingFaceEmbeddings():
    def __init__(self, question_encoder, context_encoder, question_tokenizer, context_tokenizer):
        self.model = [question_encoder,context_encoder ]
        self.question_tokenizer = question_tokenizer
        self.context_tokenizer = context_tokenizer
        

    def embed_query(self, query):
        input_ids = self.question_tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)["input_ids"]
        return (self.model[0](input_ids).pooler_output).detach().numpy()[0]

    def embed_documents(self, docs):
        encoding = []
        for d in docs:
            input_ids = self.context_tokenizer(d, return_tensors='pt', padding=True, truncation=True, max_length=512)["input_ids"]
            encoding.append((self.model[1](input_ids).pooler_output).detach().numpy()[0])
        return encoding

# Instantiate embeddings
embeddings = DPRHuggingFaceEmbeddings(question_encoder, context_encoder, question_tokenizer, context_tokenizer)


# Connecting to MILVUS vector store locally
vector_store = Milvus(
    embeddings,
    connection_args={"uri": "http://localhost:19530"},
    collection_name="customembedingncertvectorstore",
)


# Hugging Face LLM to generate answers of provided queries
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
llm = HuggingFaceEndpoint(
        repo_id=repo_id, max_length=512, temperature=0.5, token='YOUR_TOKEN'
    )

prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful assistant that that can answer questions from the provided document
        
        Answer the following question: {question}
        By searching the following document: {docs}
        
        Only use the factual information from the document to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """,
    )

chain = LLMChain(llm=llm, prompt=prompt)


# response = chain({'question' : query, 'docs' : docs_page_content})
# print(response['text'])


input=st.text_area("Enter your question here")

if input is not None:
    button=st.button("Submit")
    if button:
        docs = vector_store.similarity_search(query=input, k=5)
        docs_page_content = " ".join([d.page_content for d in docs])
        response=chain({'question' : input, 'docs' : docs_page_content})
        st.write(response['text'])
        