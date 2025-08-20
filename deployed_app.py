import os
import shutil
import asyncio
import streamlit as st
from PyPDF2 import PdfReader
from pydantic import SecretStr
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

GOOGLE_API_KEY = st.secrets["google"]["GOOGLE_API_KEY"]


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

if __name__ == "__main__":
    st.set_page_config("Multi PDF Chatbot", page_icon = ":material/unknown_5:")
    st.title("Multiple PDF RAG ChatBot")

    # Initialize session state
    if "conversations" not in st.session_state:
        st.session_state["conversations"] = {"conversation_1": []}

    if "conv_counter" not in st.session_state:
        st.session_state["conv_counter"] = 1

    if "current_conversation" not in st.session_state:
        st.session_state["current_conversation"] = "conversation_1"
    
    chat_messages_container = st.container(width=900, height=620, border=False)

    with chat_messages_container:
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


    # sidebar configuration
    with st.sidebar:
        st.title("Conversations Management")

        conv_list = list(st.session_state["conversations"].keys())

        selected_conv = st.selectbox(
            "Choose a Conversation",
            conv_list,
            index=conv_list.index(st.session_state["current_conversation"]) if st.session_state["current_conversation"] in conv_list else 0
        )
        
        if selected_conv != st.session_state["current_conversation"]:
            st.session_state["current_conversation"] = selected_conv


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
                    st.session_state["current_conversation"] = new_name
                    st.session_state["rename_flag"] = False
                    st.rerun()
        
        # delete current conversation button
        with col3:
            if st.button("🗑️", help="Delete current conversation"):
                conv_to_delete = st.session_state["current_conversation"]
                if conv_to_delete in st.session_state["conversations"]:
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
