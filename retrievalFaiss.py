from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os

def separate_situation_problem(text):
    situation_start = "Situation :"
    problem_start = "Problem:"

    situation_index = text.find(situation_start)
    problem_index = text.find(problem_start)

    if situation_index == -1 or problem_index == -1:
        return None, None  # If either part is not found

    situation = text[situation_index + len(situation_start):problem_index].strip()
    problem = text[problem_index + len(problem_start):].strip()

    return situation, problem

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details.
    If the answer is not in the provided context, just say, "Answer is not available in the context". Don't provide a wrong answer.\n\n
    Situation:\n {situation}\n
    Problem:\n{problem}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["situation", "problem"]
    )

    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question, dataPath, key):
    os.environ["GOOGLE_API_KEY"] = key

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)

    docs = new_db.similarity_search(user_question, k=3)

    chain = get_conversational_chain()

    situation, problem = separate_situation_problem(user_question)
    if situation is None or problem is None:
        return "Unable to parse the situation and problem from the input."

    response = chain(
        {"input_documents": docs, "situation": situation, "problem": problem}, return_only_outputs=True
    )

    return response["output_text"]
