from langchain_text_splitters import RecursiveCharacterTextSplitter



def chunck_data(text):

    print('Chucking The text . . .')
    chunk_size_tok = 512
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size_tok, chunk_overlap=10
    )
    texts_split = text_splitter.split_text(text)

    return texts_split

