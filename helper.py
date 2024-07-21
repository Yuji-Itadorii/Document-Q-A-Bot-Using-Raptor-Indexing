from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Dict, List, Optional, Tuple
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, TFAutoModelForSeq2SeqLM
import numpy as np
import pandas as pd
import umap
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sklearn.mixture import GaussianMixture
import os
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer


RANDOM_SEED = 224  # Fixed seed for reproducibility

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

custom_embeddings = DPRHuggingFaceEmbeddings(question_encoder, context_encoder, question_tokenizer, context_tokenizer)


# Sentence Transformer Embeddings
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embd  = HuggingFaceBgeEmbeddings(
                model_name='sentence-transformers/msmarco-bert-base-dot-v5',
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )

# Hugging Face LLM to generate Summaries of CLusters
os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'
repo_id = "Falconsai/text_summarization"
tokenizer = AutoTokenizer.from_pretrained(repo_id)
hugging_face_model = TFAutoModelForSeq2SeqLM.from_pretrained(repo_id)

pipe = pipeline(
        "summarization",
        model=hugging_face_model,
        tokenizer=tokenizer,
        max_length=128,
        truncation=True,
        batch_size=16,
    )

model = HuggingFacePipeline(pipeline=pipe)



# Funtion to extract the Text from the PDF
def extract_text(path):
    print("Etracting Text . . .")
    loader = PyPDFLoader(path)
    docs = loader.load()

    #removing extra pages
    docs = docs[:40]
    docs_texts = [d.page_content for d in docs]

    #removing special charaters
    docs_texts = [t.replace('\n', ' ') for t in docs_texts]
    docs_texts = [t.replace('\t', ' ') for t in docs_texts]
    docs_texts = [t.replace('•', ' ') for t in docs_texts]

    text = ""
    for d in docs_texts:
        text += d

    return text



# Function to Chunck the data 
def chunck_data(text):

    print('Chucking The text . . .')
    chunk_size_tok = 512
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size_tok, chunk_overlap=10
    )
    texts_split = text_splitter.split_text(text)

    return texts_split





# Function to implement RAPTOR Indexing
def implement_raptor_indexing(leaf_texts):

    print("Implementing Indexing . . .")
    
    

    def global_cluster_embeddings(
        embeddings: np.ndarray,
        dim: int,
        n_neighbors: Optional[int] = None,
        metric: str = "cosine",
    ) -> np.ndarray:
        """
        Perform global dimensionality reduction on the embeddings using UMAP.

        Parameters:
        - embeddings: The input embeddings as a numpy array.
        - dim: The target dimensionality for the reduced space.
        - n_neighbors: Optional; the number of neighbors to consider for each point.
                       If not provided, it defaults to the square root of the number of embeddings.
        - metric: The distance metric to use for UMAP.

        Returns:
        - A numpy array of the embeddings reduced to the specified dimensionality.
        """
        if n_neighbors is None:
            n_neighbors = int((len(embeddings) - 1) ** 0.5)
        return umap.UMAP(
            n_neighbors=n_neighbors, n_components=dim, metric=metric
        ).fit_transform(embeddings)


    def local_cluster_embeddings(
        embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
    ) -> np.ndarray:
        """
        Perform local dimensionality reduction on the embeddings using UMAP, typically after global clustering.

        Parameters:
        - embeddings: The input embeddings as a numpy array.
        - dim: The target dimensionality for the reduced space.
        - num_neighbors: The number of neighbors to consider for each point.
        - metric: The distance metric to use for UMAP.

        Returns:
        - A numpy array of the embeddings reduced to the specified dimensionality.
        """
        return umap.UMAP(
            n_neighbors=num_neighbors, n_components=dim, metric=metric
        ).fit_transform(embeddings)


    def get_optimal_clusters(
        embeddings: np.ndarray, max_clusters: int = 50, random_state: int = RANDOM_SEED
    ) -> int:
        """
        Determine the optimal number of clusters using the Bayesian Information Criterion (BIC) with a Gaussian Mixture     Model.

        Parameters:
        - embeddings: The input embeddings as a numpy array.
        - max_clusters: The maximum number of clusters to consider.
        - random_state: Seed for reproducibility.

        Returns:
        - An integer representing the optimal number of clusters found.
        """
        max_clusters = min(max_clusters, len(embeddings))
        n_clusters = np.arange(1, max_clusters)
        bics = []
        for n in n_clusters:
            gm = GaussianMixture(n_components=n, random_state=random_state)
            gm.fit(embeddings)
            bics.append(gm.bic(embeddings))
        return n_clusters[np.argmin(bics)]


    def GMM_cluster(embeddings: np.ndarray, threshold: float, random_state: int = 0):
        """
        Cluster embeddings using a Gaussian Mixture Model (GMM) based on a probability threshold.

        Parameters:
        - embeddings: The input embeddings as a numpy array.
        - threshold: The probability threshold for assigning an embedding to a cluster.
        - random_state: Seed for reproducibility.

        Returns:
        - A tuple containing the cluster labels and the number of clusters determined.
        """
        n_clusters = get_optimal_clusters(embeddings)
        gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
        gm.fit(embeddings)
        probs = gm.predict_proba(embeddings)
        labels = [np.where(prob > threshold)[0] for prob in probs]
        return labels, n_clusters


    def perform_clustering(
        embeddings: np.ndarray,
        dim: int,
        threshold: float,
    ) -> List[np.ndarray]:
        """
        Perform clustering on the embeddings by first reducing their dimensionality globally, then clustering
        using a Gaussian Mixture Model, and finally performing local clustering within each global cluster.

        Parameters:
        - embeddings: The input embeddings as a numpy array.
        - dim: The target dimensionality for UMAP reduction.
        - threshold: The probability threshold for assigning an embedding to a cluster in GMM.

        Returns:
        - A list of numpy arrays, where each array contains the cluster IDs for each embedding.
        """
        if len(embeddings) <= dim + 1:
            # Avoid clustering when there's insufficient data
            return [np.array([0]) for _ in range(len(embeddings))]

        # Global dimensionality reduction
        reduced_embeddings_global = global_cluster_embeddings(embeddings, dim)
        # Global clustering
        global_clusters, n_global_clusters = GMM_cluster(
            reduced_embeddings_global, threshold
        )

        all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
        total_clusters = 0

        # Iterate through each global cluster to perform local clustering
        for i in range(n_global_clusters):
            # Extract embeddings belonging to the current global cluster
            global_cluster_embeddings_ = embeddings[
                np.array([i in gc for gc in global_clusters])
            ]

            if len(global_cluster_embeddings_) == 0:
                continue
            if len(global_cluster_embeddings_) <= dim + 1:
                # Handle small clusters with direct assignment
                local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
                n_local_clusters = 1
            else:
                # Local dimensionality reduction and clustering
                reduced_embeddings_local = local_cluster_embeddings(
                    global_cluster_embeddings_, dim
                )
                local_clusters, n_local_clusters = GMM_cluster(
                    reduced_embeddings_local, threshold
                )

            # Assign local cluster IDs, adjusting for total clusters already processed
            for j in range(n_local_clusters):
                local_cluster_embeddings_ = global_cluster_embeddings_[
                    np.array([j in lc for lc in local_clusters])
                ]
                indices = np.where(
                    (embeddings == local_cluster_embeddings_[:, None]).all(-1)
                )[1]
                for idx in indices:
                    all_local_clusters[idx] = np.append(
                        all_local_clusters[idx], j + total_clusters
                    )

            total_clusters += n_local_clusters

        return all_local_clusters


    ### --- Our code below --- ###


    def embed(texts):
        """
        Generate embeddings for a list of text documents.

        This function assumes the existence of an `embd` object with a method `embed_documents`
        that takes a list of texts and returns their embeddings.

        Parameters:
        - texts: List[str], a list of text documents to be embedded.

        Returns:
        - numpy.ndarray: An array of embeddings for the given text documents.
        """
        text_embeddings = embd.embed_documents(texts)
        text_embeddings_np = np.array(text_embeddings)
        return text_embeddings_np


    def embed_cluster_texts(texts):
        """
        Embeds a list of texts and clusters them, returning a DataFrame with texts, their embeddings, and cluster   labels.

        This function combines embedding generation and clustering into a single step. It assumes the existence
        of a previously defined `perform_clustering` function that performs clustering on the embeddings.

        Parameters:
        - texts: List[str], a list of text documents to be processed.

        Returns:
        - pandas.DataFrame: A DataFrame containing the original texts, their embeddings, and the assigned cluster   labels.
        """
        text_embeddings_np = embed(texts)  # Generate embeddings
        cluster_labels = perform_clustering(
            text_embeddings_np, 10, 0.1
        )  # Perform clustering on the embeddings
        df = pd.DataFrame()  # Initialize a DataFrame to store the results
        df["text"] = texts  # Store original texts
        df["embd"] = list(text_embeddings_np)  # Store embeddings as a list in the DataFrame
        df["cluster"] = cluster_labels  # Store cluster labels
        return df


    def fmt_txt(df: pd.DataFrame) -> str:
        """
        Formats the text documents in a DataFrame into a single string.

        Parameters:
        - df: DataFrame containing the 'text' column with text documents to format.

        Returns:
        - A single string where all text documents are joined by a specific delimiter.
        """
        unique_txt = df["text"].tolist()
        return "--- --- \n --- --- ".join(unique_txt)


    def embed_cluster_summarize_texts(
        texts: List[str], level: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Embeds, clusters, and summarizes a list of texts. This function first generates embeddings for the texts,
        clusters them based on similarity, expands the cluster assignments for easier processing, and then summarizes
        the content within each cluster.

        Parameters:
        - texts: A list of text documents to be processed.
        - level: An integer parameter that could define the depth or detail of processing.

        Returns:
        - Tuple containing two DataFrames:
          1. The first DataFrame (`df_clusters`) includes the original texts, their embeddings, and cluster assignments.
          2. The second DataFrame (`df_summary`) contains summaries for each cluster, the specified level of detail,
             and the cluster identifiers.
        """

        # Embed and cluster the texts, resulting in a DataFrame with 'text', 'embd', and 'cluster' columns
        df_clusters = embed_cluster_texts(texts)

        # Prepare to expand the DataFrame for easier manipulation of clusters
        expanded_list = []

        # Expand DataFrame entries to document-cluster pairings for straightforward processing
        for index, row in df_clusters.iterrows():
            for cluster in row["cluster"]:
                expanded_list.append(
                    {"text": row["text"], "embd": row["embd"], "cluster": cluster}
                )

        # Create a new DataFrame from the expanded list
        expanded_df = pd.DataFrame(expanded_list)

        # Retrieve unique cluster identifiers for processing
        all_clusters = expanded_df["cluster"].unique()

        print(f"--Generated {len(all_clusters)} clusters--")

        # Summarization
        template = "{context}"
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | model | StrOutputParser()

        # Format text within each cluster for summarization
        summaries = []
        count = 1
        for i in all_clusters:
          print(count)
          count = count+1
          df_cluster = expanded_df[expanded_df["cluster"] == i]
          formatted_txt = fmt_txt(df_cluster)
        #   print(formatted_txt)
          summaries.append(chain.invoke({"context": formatted_txt}))
          print(summaries[-1])

        # Create a DataFrame to store summaries with their corresponding cluster and level
        df_summary = pd.DataFrame(
            {
                "summaries": summaries,
                "level": [level] * len(summaries),
                "cluster": list(all_clusters),
            }
        )

        return df_clusters, df_summary


    def recursive_embed_cluster_summarize(
        texts: List[str], level: int = 1, n_levels: int = 3
    ) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]:

        results = {}  # Dictionary to store results at each level

        # Perform embedding, clustering, and summarization for the current level
        df_clusters, df_summary = embed_cluster_summarize_texts(texts, level)

        # Store the results of the current level
        results[level] = (df_clusters, df_summary)

        # Determine if further recursion is possible and meaningful
        unique_clusters = df_summary["cluster"].nunique()
        print(f"Unique clusters at level {level}: {unique_clusters}")
        if level < n_levels and unique_clusters > 1:
            # Use summaries as the input texts for the next level of recursion
            new_texts = df_summary["summaries"].tolist()
            # print(new_texts)
            # print(len(new_texts))
            # for n in new_texts:
            #     print(num_tokens_from_string(n , "cl100k_base"))
            next_level_results = recursive_embed_cluster_summarize(
                new_texts, level + 1, n_levels
            )

            # Merge the results from the next level into the current results dictionary
            results.update(next_level_results)

        return results
    
    results = recursive_embed_cluster_summarize(leaf_texts, level=1, n_levels=10)

    return results



# Function to create a Milvus Vector Store to store the generated embedding after implementing RAPTOR indexing
def create_vector_store(leaf_texts , results):
    print('Creating a vector store . . . ')
    
    # Initialize all_texts with leaf_texts
    all_texts = leaf_texts.copy()


    # Iterate through the results to extract summaries from each level and add them to all_texts
    for level in sorted(results.keys()):
        # Extract summaries from the current level's DataFrame
        summaries = results[level][1]["summaries"].tolist()
        # Extend all_texts with the summaries from the current level
        all_texts.extend(summaries)

    # Now, use all_texts to build the vectorstore with Chroma
    vectorstore = Milvus.from_texts(
        texts=all_texts,
        embedding=custom_embeddings,
        collection_name="customembedingncertvectorstore",
        connection_args={"host": "localhost", "port": 19530},
    )

    return vectorstore
