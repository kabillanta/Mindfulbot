import streamlit as st
from utils import send_otp, validate_otp, correct_otp

# Streamlit Login Page
def main():
    st.title("Login - MindfulBot")
    st.write("Welcome! Please log in to continue.")

    # Email Input
    email = st.text_input("Enter your email:")
    if st.button("Send OTP") and email:
        send_otp(email)

    # OTP Validation
    otp_input = st.text_input("Enter OTP:", type="password")
    if st.button("Validate OTP"):
        if validate_otp(otp_input, correct_otp):
            st.success("Login successful!")
            st.write("Redirecting to chat...")
            st.query_params(page="chat")
            st.experimental_rerun()  # Redirect to chat in Flask

if __name__ == "__main__":
    main()
