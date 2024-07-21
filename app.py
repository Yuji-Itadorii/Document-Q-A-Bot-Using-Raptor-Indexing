

from helper import extract_text,chunck_data,implement_raptor_indexing, create_vector_store



path = 'Data/NCRT.pdf'

text = extract_text(path=path)

text_splits = chunck_data(text=text)

results = implement_raptor_indexing(leaf_texts=text_splits)

db = create_vector_store(leaf_texts=text_splits, results=results)

