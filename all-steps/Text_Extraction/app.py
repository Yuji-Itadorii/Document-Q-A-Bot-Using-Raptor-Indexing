from langchain_community.document_loaders import PyPDFLoader

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