# from retrievalChroma import retreival_response
from retrievalFaiss import user_input
import streamlit as st
import time
import os



# Simulate a function that generates a response with a delay
def generate_response(user_message,key):
    dbpath = 'faiss_db'
    response = user_input(user_message, dbpath,key)
    
    time.sleep(2)  # Simulating a delay for response generation
    return f" {response}"



# Below Frontend

# Initialize session state for API key and messages
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'Rag' not in st.session_state:
    st.session_state.Rag = []
# if 'new' not in st.session_state:
#     st.session_state.new = False
if 'page' not in st.session_state:
    st.session_state.page = "Home"

# Function to display chat interface
def chat_interface():

        # Display chat history
    for chat in st.session_state.Rag:
        if chat["role"] == "user":
            with st.chat_message("user"):
                st.markdown(f"**You:** {chat['content']}")
        else:
            with st.chat_message("assistant"):
                st.markdown(f"**Assistant:** {chat['content']}")

    # Accept user input
    prompt = st.chat_input("Say something")

    if prompt:

        # Add user message to chat history
        st.session_state.Rag.append({"role": "user", "content": prompt})

        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(f"**You:** {prompt}")

        # Display loading message
        loading_message_placeholder = st.empty()
        loading_message_placeholder.markdown("**Loading...**")

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            response = generate_response(prompt,st.session_state.api_key)
            st.markdown(f"**Assistant:** {response}")

        # Clear loading message and display response
        loading_message_placeholder.empty()
        st.session_state.Rag.append({"role": "assistant", "content": response})

    # st.write(messages_key)

# Main application
st.set_page_config(page_title="QA RAG App", layout="wide")

# Sidebar for navigation
if st.session_state.api_key:
    
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Go to", ["Home", "QA Chat"])
else:
    # st.session_state.new = False
    page = "Home"

if page == "Home":
    st.title("Home Page")
    st.markdown("""
        ## Welcome to the QA RAG App!

        This app demonstrates a Retrieval-Augmented Generation (RAG) implementation 
        using Gemini API with Chroma and Faiss databases. 
        It is part of an internship project.

        ### How to Use
        1. Enter your Gemini API key below.
        2. Navigate to the chat pages using the sidebar.
        3. Start interacting with the QA system.

        ### Watch the Video
        Below is a video that explains the basics of RAG and how it works:

        <iframe width="560" height="315" src="https://www.youtube.com/embed/dQw4w9WgXcQ" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

        ### Get Your Gemini API Key
        Don't have a Gemini API key? [Get your API key here](https://support.gemini.com/hc/en-us/articles/360031080191-How-do-I-create-an-API-key#:~:text=API%20keys%20can%20be%20created,use%20when%20creating%20API%20keys%3F)
    """, unsafe_allow_html=True)

    # Input to save API key
    api_key_input = st.text_input("Enter Gemini API key:", type="password")
    if st.button("Save API Key"):
        st.session_state.api_key = api_key_input
        st.session_state.page = "QA Chat"
        os.environ["GOOGLE_API_KEY"]=api_key_input
        # st.session_state.new = True
        st.success("API key saved! You can now access the chat pages from the sidebar.")
        st.experimental_rerun()  # Rerun to update the page state

    if st.session_state.api_key:
        if st.button("Reset API Key"):
            st.session_state.api_key = ""
            st.session_state.page = "Home"
            st.success("API key reset! Please enter a new API key on the Home page.")
            st.experimental_rerun()  # Rerun to update the page state

elif page == "QA Chat":
    st.title("Rag Implementation")
    st.session_state.page = "Rag"
    # st.session_state.new = False
    chat_interface()


    
