from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
 
# Creating an UnstructuredFileLoader to load text.
loader = UnstructuredFileLoader("The_Nightingale.txt")

# Loading raw documents using document_loaders from LangChain
raw_documents = loader.load()
print("Document loaded using document_loaders from LangChain")

# Initializing a CharacterTextSplitter for text splitting
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
)


# Splitting text into documents using text_splitter from LangChain
documents = text_splitter.split_documents(raw_documents)
print("Text splitted using text_splitter from LangChain")

# Initializing OpenAIEmbeddings for text embeddings
embeddings = OpenAIEmbeddings()

# Creating a vectorstore using FAISS from LangChain
vectorstore = FAISS.from_documents(documents, embeddings)
print("created a vectorestore using FAISS from LangChain")


# Saving the vectorstore locally
vectorstore.save_local("faiss_index_constitution")
print("Vectorestore saved in 'faiss_index_constitution' ")

