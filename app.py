import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Initialize API and Embeddings
@st.cache_resource
def get_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory="db/chroma_db",  
    )
    return vector_store

@st.cache_resource
def get_groq_client():
    return Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )

vector_store = get_vector_store()
client = get_groq_client()

def ReformulateQuery(user_query, chat_history):
    # Flatten history into a readable string
    history_str = ""
    for idx, exchange in chat_history.items():
        history_str += f"User: {exchange['query']}\nBot: {exchange['response']}\n"
        
    prompt = f"""Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
    User query: {user_query}
    Chat history: {history_str}"""
    
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model="llama-3.3-70b-versatile",
    )
    return chat_completion.choices[0].message.content

# Set up page configuration
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="centered")

# UI design
st.title("📚 History-Aware RAG Assistant")
st.caption("Ask me anything! I have access to your documents and can remember our previous messages.")
st.divider()

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}
    st.session_state.cnt = 1

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Enter your query..."):
    # Add user message to chat UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            user_query = prompt
            chat_history = st.session_state.chat_history
            
            reformulated_query = user_query
            if len(chat_history) != 0:
                reformulated_query = ReformulateQuery(user_query, chat_history)

            results = vector_store.similarity_search(reformulated_query, k=10)
            context_for_llm = "\n\n".join([doc.page_content for doc in results])
            
            llm_prompt = f"""
            Answer the question based on the given context.
            Context: {context_for_llm}
            Question: {reformulated_query}
            """
            
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "user", "content": llm_prompt}
                ],
                model="llama-3.3-70b-versatile",
            )
            
            response = chat_completion.choices[0].message.content
            st.markdown(response)
            
            # Save assistant response to UI chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Save to underlying structured history
            st.session_state.chat_history[st.session_state.cnt] = {
                'query': reformulated_query,
                'response': response
            }
            st.session_state.cnt += 1
