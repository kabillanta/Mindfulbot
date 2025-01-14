from flask import Flask, render_template, request, redirect, url_for
import os

app = Flask(__name__)

# Route for login
@app.route('/')
def login():
    return redirect("/streamlit-login")

# Route for chat
@app.route('/chat')
def chat():
    return redirect("/streamlit-chat")

# Embedding Streamlit
@app.route('/streamlit-login')
def streamlit_login():
    os.system("streamlit run login.py")
    return "Streamlit Login page running..."  # Optional message

@app.route('/streamlit-chat')
def streamlit_chat():
    os.system("streamlit run chat.py")
    return "Streamlit Chat page running..."  # Optional message

if __name__ == "__main__":
    app.run(debug=True)
