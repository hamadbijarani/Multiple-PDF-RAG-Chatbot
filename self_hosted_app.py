import os
import shutil
import sqlite3
import asyncio
import streamlit as st
from PyPDF2 import PdfReader
from pydantic import SecretStr
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.callbacks.base import BaseCallbackHandler
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

DB_PATH = "conversations.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()

c.execute("""
    CREATE TABLE IF NOT EXISTS chats (
        conv_id TEXT,
        role TEXT,
        content TEXT
    )
""")
conn.commit()


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=2000)
    text_chunks = splitter.split_text(text)
    return text_chunks

def get_pdf_text(pdf_docs):
    text = ""
    for docs in pdf_docs:
        pdf = PdfReader(docs)
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

    return text

def get_vector_store(text_chunks):
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable is missing.")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=SecretStr(GOOGLE_API_KEY))

    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversation_chain():
    prompt = """
        You are an AI assistant designed to answer questions with or without the provided context. 
        Follow these rules carefully:

        1. **Context Usage**
        - If the question is related to the provided context, use the context to answer as accurately as possible.
        - Focus on meaning and intent, not just keyword matches.
        - If the context is insufficient or incomplete, you may add general knowledge to fill the gap but clearly prioritize the context.

        2. **Out-of-Context Handling**
        - If the question is unrelated to the context, ignore the context and answer naturally using your own knowledge.

        3. **Answer Style**
        - Respond in a clear, natural, and human-friendly tone (like a helpful chatbot).
        - Keep answers informative, structured, and to the point.
        - Adapt response length to match user preferences inferred from the conversation history.
        - Avoid hallucinating details when relying on context.

        4. **Conversation Continuity**
        - Stay polite, engaging, and approachable.
        - If the user asks follow-up questions, smoothly carry the conversation forward.

        ---

        Context:
        {context}

        Question:
        {question}

        Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, api_key=GOOGLE_API_KEY, streaming=True)
    prompt = PromptTemplate(template=prompt, input_variables=['context', 'question'])

    chain = load_qa_chain(model, prompt=prompt, chain_type="stuff")

    return chain

def get_user_prompt(user_input):
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable is missing.")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=SecretStr(GOOGLE_API_KEY))

    faiss_dir = "faiss_index"
    index_file = os.path.join(faiss_dir, "index.faiss")

    if os.path.exists(index_file):
        vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = vector_db.similarity_search(user_input)

        chain = get_conversation_chain()

        response = chain(
            {"input_documents":docs, "question": user_input},
            return_only_outputs=True )

        return response.get("output_text", "Sorry, I couldn't find an answer.")
    else:
        model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, api_key=GOOGLE_API_KEY, streaming=True)
        response = model.invoke(user_input)
        return response.content

def save_current_conversation():
    conv_id = st.session_state["current_conversation"]
    messages = st.session_state["conversations"][conv_id]
    c.execute("DELETE FROM chats WHERE conv_id=?", (conv_id,))
    for msg in messages:
        c.execute("INSERT INTO chats VALUES (?, ?, ?)", (conv_id, msg["role"], msg["content"]))
    conn.commit()

def delete_conversation(conv_id):
    c.execute("DELETE FROM chats WHERE conv_id=?", (conv_id,))
    conn.commit()

def rename_conversation(old_name, new_name):
    c.execute("""UPDATE chats SET conv_id=? WHERE conv_id=?""", (new_name, old_name,))
    conn.commit()

def load_all_conversations():
    """Return all conversation IDs."""
    c.execute("SELECT DISTINCT conv_id FROM chats")
    return [row[0] for row in c.fetchall()]

def load_conversation(conv_id):
    """Load one conversation's messages."""
    c.execute("SELECT role, content FROM chats WHERE conv_id=?", (conv_id,))
    return [{"role": r, "content": c} for r, c in c.fetchall()]

if __name__ == "__main__":
    st.set_page_config("Multi PDF Chatbot", page_icon = ":material/unknown_5:")
    st.title("Multiple PDF RAG ChatBot")

    # initialize chat history
    all_convs = load_all_conversations()

    # Initialize session state
    if "conversations" not in st.session_state:
        st.session_state["conversations"] = {}

    if "conv_counter" not in st.session_state:
        if all_convs:
            max_num = 0
            for conv in all_convs:
                if conv.startswith("conversation_"):
                    try:
                        num = int(conv.split("_")[1])
                        max_num = max(max_num, num)
                    except (IndexError, ValueError):
                        pass
            st.session_state["conv_counter"] = max_num
        else:
            st.session_state["conv_counter"] = 0

    if "current_conversation" not in st.session_state:
        if all_convs:
            st.session_state["current_conversation"] = all_convs[0]
        else:
            st.session_state["current_conversation"] = "conversation_1"
            st.session_state["conversations"] = {"conversation_1": []}
            if st.session_state["conv_counter"] == 0:
                st.session_state["conv_counter"] = 1
    
    chat_messages_container = st.container(width=900, height=620, border=False)

    with chat_messages_container:
        # load current conversation if not already loaded in memory
        current_conv_id = st.session_state["current_conversation"]
        if current_conv_id not in st.session_state["conversations"]:
            st.session_state["conversations"][current_conv_id] = load_conversation(current_conv_id)
        
        # display chat history
        messages = st.session_state["conversations"][st.session_state["current_conversation"]]

        for message in messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # get prompt from the user
    ques = st.chat_input("What is up?")

    with chat_messages_container:
        if prompt := ques:
            # save user message
            messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # get llm response
            try:
                response = get_user_prompt(prompt)
                
                # save llm message
                messages.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
            except Exception as e:
                error_message = f"Error: {str(e)}"
                messages.append({"role": "assistant", "content": error_message})
                with st.chat_message("assistant"):
                    st.markdown(error_message)
            save_current_conversation()


    # sidebar configuration
    with st.sidebar:
        st.title("Conversations Management")

        conv_list = load_all_conversations()
        if not conv_list:
            conv_list = ["conversation_1"]

        if st.session_state["current_conversation"] not in conv_list:
            conv_list.append(st.session_state["current_conversation"])

        selected_conv = st.selectbox(
            "Choose a Conversation",
            conv_list,
            index=conv_list.index(st.session_state["current_conversation"]) if st.session_state["current_conversation"] in conv_list else 0
        )
        
        if selected_conv != st.session_state["current_conversation"]:
            st.session_state["current_conversation"] = selected_conv
            if selected_conv not in st.session_state["conversations"]:
                st.session_state["conversations"][selected_conv] = load_conversation(selected_conv)


        col1, col2, col3 = st.columns(3)

        # new conversation button
        with col1: 
            if st.button("➕", help="New Conversation"):
                st.session_state["conv_counter"] += 1
                new_conv_id = f"conversation_{st.session_state['conv_counter']}"
                st.session_state["conversations"][new_conv_id] = []
                st.session_state["current_conversation"] = new_conv_id
                st.rerun()

        # rename current conversation button
        with col2:
            if "rename_flag" not in st.session_state:
                st.session_state["rename_flag"] = False

            if st.button("🖊️", help="Rename current conversation"):
                st.session_state["rename_flag"] = True

            if st.session_state["rename_flag"]:
                new_name = st.text_input("Enter new conversation name:", value=st.session_state["current_conversation"])
                if new_name and new_name != st.session_state["current_conversation"]:
                    st.session_state["conversations"][new_name] = st.session_state["conversations"].pop(st.session_state["current_conversation"])
                    rename_conversation(st.session_state["current_conversation"], new_name)
                    st.session_state["current_conversation"] = new_name
                    st.session_state["rename_flag"] = False
                    st.rerun()
        
        # delete current conversation button
        with col3:
            if st.button("🗑️", help="Delete current conversation"):
                conv_to_delete = st.session_state["current_conversation"]
                if conv_to_delete in st.session_state["conversations"]:
                    delete_conversation(conv_to_delete)
                    st.session_state["conversations"].pop(conv_to_delete)

                    # load default conversation
                    st.session_state["conv_counter"] += 1
                    new_conv_id = f"conversation_{st.session_state['conv_counter']}"
                    st.session_state["conversations"][new_conv_id] = []
                    st.session_state["current_conversation"] = new_conv_id
                    st.rerun()

        
        st.markdown("---")

        st.title("Upload PDFs")

        pdf_expander = st.expander("PDF files")

        with pdf_expander:
            pdf_docs = st.file_uploader(
                "Upload PDF Files",
                accept_multiple_files=True,
                type="pdf"
            )

        if st.button("ReProcess"):
            with st.spinner("Processing..."):
                if pdf_docs:
                    raw_text = get_pdf_text(pdf_docs)       # extract text
                    text_chunks = get_text_chunks(raw_text) # split into chunks
                    get_vector_store(text_chunks)           # create vector store
                    st.success("Done!")
                else:
                    if os.path.exists("faiss_index"):
                        shutil.rmtree("faiss_index")
                    st.warning("No PDFs uploaded")

        st.write("\nDeveloped by: **Hamadullah Bijarani**")
