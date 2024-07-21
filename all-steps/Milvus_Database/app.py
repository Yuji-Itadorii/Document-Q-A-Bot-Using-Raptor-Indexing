from langchain_community.vectorstores import Milvus


# Note: embd is the embeddings that we have to initialize by huggingface embegging library



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
        embedding=embd,
        collection_name="ncert-vectorstore",
        connection_args={"host": "localhost", "port": 19530},
    )

    return vectorstore
