import streamlit as st
import random
import smtplib
from email.mime.text import MIMEText
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer, util
import numpy as np
import PyPDF2
from io import BytesIO

# Imports for RAG and FAISS-CPU
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS

# Initialize Streamlit app
st.title("MindfulBot\nYour Mental Health Support Companion")

# Add disclaimer
st.markdown("""
    **Important Disclaimer:** This bot is for informational purposes only and is not a substitute for professional mental health care. 
    If you're experiencing a crisis or emergency, please contact emergency services or crisis helpline immediately.
    
    National Crisis Hotline (US): 988
""")

# Function to send OTP via email
def send_otp(email):
    otp = random.randint(100000, 999999)
    st.session_state.otp = otp

    sender_email = "mindfulbot@gmail.com"  # Replace with your email
    sender_password = ""  # Replace with your app-specific password
    smtp_server = "smtp.gmail.com"
    smtp_port = 587

    subject = "Your Secure Access Code"
    body = f"Your secure access code for MindfulBot is: {otp}\n\nYour privacy and security are important to us."
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = email

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, email, msg.as_string())
        server.quit()
        st.success(f"Secure code sent to {email}")
    except Exception as e:
        st.error(f"Failed to send secure code: {e}")

# Function to validate the OTP
def validate_otp(user_otp):
    if int(user_otp) == st.session_state.otp:
        st.session_state.authenticated = True
        st.success("Access granted. Welcome to your safe space.")
        st.rerun()
    else:
        st.error("Invalid code. Please try again.")

# Function to check if the question is related to mental health topics
def is_relevant_question(question, embedding_model, category_embeddings):
    threshold = 0.01
    question_embedding = embedding_model.encode(question, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(question_embedding, category_embeddings)
    similarities_np = similarities.numpy()
    max_index = np.argmax(similarities_np)
    max_similarity = similarities_np.flatten()[max_index]
    return max_similarity > threshold

# Function to append conversation history
def append_to_history(user_input, ai_response):
    if 'history' not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append({'user_input': user_input, 'ai_response': ai_response})

# Function to read PDF files (for mental health resources)
def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to filter responses for supportive language
def filter_response(response):
    # Implement content filtering and ensure supportive language
    # Add trigger warning if needed
    return response

# Initialize LangChain memory
memory = ConversationBufferWindowMemory(k=5)

# Groq API key
groq_api_key = "api key"  

if groq_api_key:
    model_name = "llama3-70b-8192"
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model_name
    )

    # Define mental health focused prompt template
    template = """You are MindfulBot, a compassionate mental health support companion. Your primary functions are:
    1. Provide emotional support and understanding in a non-judgmental way
    2. Share information about mental health, wellness, and coping strategies
    3. Help users explore their feelings and thoughts safely
    4. Guide users to professional resources when appropriate
    5. Maintain strict confidentiality and privacy
    6. Never provide medical advice or diagnosis
    7. Always include crisis resources when discussing serious topics
    8. Use trauma-informed and person-centered language
    
    Remember: If a user expresses thoughts of self-harm or suicide, immediately provide crisis resources and encourage professional help.

    Current conversation:
    {history}
    Human: {input}
    AI Assistant: """

    prompt = PromptTemplate(input_variables=["history", "input"], template=template)

    conversation = ConversationChain(
        llm=groq_chat,
        memory=memory,
        prompt=prompt
    )

    # Initialize Sentence Transformer model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Mental health related keywords
    mental_health_keywords = [
        # Emotional Support
        "anxiety", "depression", "stress",
        "feelings", "emotions", "mood",
        "self-care", "wellness", "mindfulness",
        "coping", "support", "therapy",
        
        # Mental Health Topics
        "mental health", "counseling",
        "meditation", "relaxation",
        "trauma", "healing", "recovery",
        "self-esteem", "boundaries", "growth",
        
        # Wellness Activities
        "exercise", "sleep", "nutrition",
        "journaling", "breathing", "grounding",
        "support group", "community", "connection",
        "mindfulness", "meditation", "yoga"
    ]

    category_embeddings = embedding_model.encode(mental_health_keywords, convert_to_tensor=True)

    # Initialize RAG components for mental health resources
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = FAISS.from_texts([""], embeddings)

    # Secure authentication
    if not st.session_state.get('authenticated', False):
        st.markdown("### Secure Access")
        st.write("Your privacy is important to us. Please verify your email to begin.")
        
        email = st.text_input("Enter your email for secure access:")
        if st.button("Send Secure Code") and email:
            send_otp(email)

        otp_input = st.text_input("Enter your secure code:", type="password")
        if st.button("Verify Access") and otp_input:
            validate_otp(otp_input)
    else:
        # Main interface after authentication
        st.write("🌟 Welcome to your safe space. How can I support you today?")

        # Support resources uploader
        uploaded_file = st.file_uploader("Upload mental health resources (PDF)", type=["pdf"])
        if uploaded_file is not None:
            file_content = read_pdf(uploaded_file)
            chunks = text_splitter.split_text(file_content)
            new_vectorstore = FAISS.from_texts(chunks, embeddings)
            st.session_state.vectorstore.merge_from(new_vectorstore)
            st.success("Resources added successfully!")

        # Tabs for different types of support
        tab1, tab2 = st.tabs(["Emotional Support", "Resource Library"])

        with tab1:
            st.subheader("Share what's on your mind")
            user_input = st.text_area("I'm here to listen:", help="Express yourself freely in a safe, confidential space")
            if user_input:
                if is_relevant_question(user_input, embedding_model, category_embeddings):
                    response = conversation.predict(input=user_input)
                    filtered_response = filter_response(response)
                    append_to_history(user_input, filtered_response)
                    st.text_area("MindfulBot:", value=filtered_response, height=200, disabled=True)
                else:
                    st.write("I want to make sure I provide the most appropriate support. Could you share more about how this relates to your mental health and wellbeing?")

        with tab2:
            st.subheader("Access Mental Health Resources")
            resource_question = st.text_area("Ask about available resources:")
            if resource_question:
                qa_chain = RetrievalQA.from_chain_type(
                    llm=groq_chat,
                    chain_type="stuff",
                    retriever=st.session_state.vectorstore.as_retriever()
                )
                resource_response = qa_chain(resource_question)
                st.text_area("Available Resources:", value=resource_response['result'], height=200, disabled=True)

        # Show conversation history in a caring way
        if 'history' in st.session_state and st.session_state.history:
            with st.expander("📝 Your Journey (Previous Conversations)"):
                for idx, interaction in enumerate(reversed(st.session_state.history)):
                    st.markdown(f"**You:** {interaction['user_input']}")
                    st.markdown(f"**MindfulBot:** {interaction['ai_response']}")
                    if idx < len(st.session_state.history) - 1:
                        st.markdown("---")

        # Always visible crisis resources
        st.sidebar.markdown("""
        ### 24/7 Crisis Support
        - 🆘 **Emergency:** 911
        - 🤝 **988 Lifeline:** Call or text 988
        - 💬 **Crisis Text Line:** Text HOME to 741741
        
        Remember: You're not alone. Help is always available.
        """)