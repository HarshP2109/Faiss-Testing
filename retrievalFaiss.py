from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
# import google.generativeai as genai
from langchain.prompts import PromptTemplate
import os


# def getPath(foldName):    
#     current_directory = os.getcwd()
#     database_directory = os.path.join(current_directory, 'Database')
#     database_directory = os.path.join(current_directory, foldName)
#     # index_faiss_path = os.path.join(database_directory, 'index.faiss')
#     return database_directory


def get_conversational_chain():
    # Define a prompt template for asking questions based on a given context
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details,
    if the answer is not in the provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    # Initialize a ChatGoogleGenerativeAI model for conversational AI
    # model = ChatVertexAI(model="gemini-pro", temperature=0.3)
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

    # Create a prompt template with input variables "context" and "question"
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Load a question-answering chain with the specified model and prompt
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question, dataPath,key):

    # newPath = getPath(dataPath)
    # index_path = os.path.join(dataPath, "index.faiss")
    os.environ["GOOGLE_API_KEY"]=key
    # if not os.path.exists(index_path):
    #     raise FileNotFoundError(f"FAISS index file not found at path: {index_path}")
    # Create embeddings for the user question using a Google Generative AI model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Load a FAISS vector database from a local file
    # new_db = FAISS.load_local("/faiss_db", embeddings, allow_dangerous_deserialization=True)
    new_db = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)

    # Perform similarity search in the vector database based on the user question
    # docs = new_db.similarity_search(user_question)
    docs = new_db.similarity_search(user_question, k=3)

    # Obtain a conversational question-answering chain
    chain = get_conversational_chain()

    # Use the conversational chain to get a response based on the user question and retrieved documents
    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True
    )

    # Print the response to the console
    # print(response["output_text"])
    return response["output_text"]