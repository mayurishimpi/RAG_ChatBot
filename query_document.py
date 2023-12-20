from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS


template = """You are an AI assistant for answering questions about Kristin Hannah's bestselling novels, "The Great Alone" and "The Nightingale". 
Provide concise and direct answers to the questions.
Question: {question}
=========
{context}
=========
Answer in Markdown:"""

# Defining a PromptTemplate for question-answering prompt
QA_PROMPT = PromptTemplate(template=template, input_variables=[
                           "question", "context"])


def load_retriever():
    # Initializing OpenAIEmbeddings for text embeddings
    embeddings = OpenAIEmbeddings()
    
    # Load the FAISS index named "faiss_index_constitution" that was store locally.
    vectorstore = FAISS.load_local("faiss_index_constitution", embeddings)
    retriever = VectorStoreRetriever(vectorstore=vectorstore)
    return retriever


def qa_chain():
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)

    # Loading the retriever for question-answering
    retriever = load_retriever()

    # Initializing a ConversationBufferMemory for storing conversation history
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True) 
    
    # Creating a ConversationalRetrievalChain model from language model, retriever, and memory
    model = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT})
    return model

