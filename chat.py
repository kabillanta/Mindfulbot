import streamlit as st
from utils import filter_response, append_to_history, read_pdf

# Chat Page
def main():
    st.title("Chat - MindfulBot")
    st.write("Welcome to your safe space.")

    # Chat Interface
    user_input = st.text_area("Share what's on your mind:")
    if user_input:
        # Simulate AI response
        response = f"Thank you for sharing. Here's some guidance: {user_input}"
        filtered_response = filter_response(response)
        append_to_history(user_input, filtered_response)
        st.text_area("MindfulBot:", value=filtered_response, height=200, disabled=True)

if __name__ == "__main__":
    main()
