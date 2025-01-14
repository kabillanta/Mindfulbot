import random
from email.mime.text import MIMEText
import smtplib

correct_otp = random.randint(100000, 999999)  # Generate a random 6-digit OTP

def send_otp(email):
    # Replace with your email credentials
    sender_email = "mithun3130@gmail.com"  # Replace with your email
    sender_password = "eypr ejsh ueol mhwi"  # Replace with your app-specific password      
    smtp_server = "smtp.gmail.com"
    smtp_port = 587

    subject = "Your OTP"
    body = f"Your OTP is: {correct_otp}"

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
        print(f"OTP sent to {email}")
        return correct_otp  # Return the generated OTP ######################################
    except Exception as e:
        print(f"Error: {e}")
        return None

# Append to history
def append_to_history(user_input, response):
    # Append conversation history to a file or database (placeholder implementation)
    with open("chat_history.txt", "a") as file:
        file.write(f"User: {user_input}\n")
        file.write(f"Bot: {response}\n")
    print("Chat history updated.")

# Filter response
def filter_response(response):
    # Modify responses to ensure supportive language (placeholder implementation)
    return response.replace("problem", "challenge").replace("difficult", "tricky")

# Generate response for user input
def generate_response(user_input):
    # Placeholder for a more complex response generation logic
    responses = [
        "I'm here to listen.",
        "It's okay to feel this way.",
        "Tell me more about what's on your mind.",
        "I'm here for you.",
        "You are not alone in this."
    ]
    return random.choice(responses)

def validate_otp(user_otp, otp):
    try:
        return int(user_otp) == int(otp)
    except ValueError:
        return False

# Example usage
if __name__ == "__main__":
    email = input("Enter your email: ")
    correct_otp = send_otp(email)
    
    if correct_otp:
        user_otp = input("Enter the OTP sent to your email: ")
        if validate_otp(user_otp, correct_otp):
            print("OTP validated successfully!")
            while True:
                user_input = input("What's on your mind? (Type 'exit' to quit): ")
                if user_input.lower() == 'exit':
                    break
                response = generate_response(user_input)
                response = filter_response(response)
                append_to_history(user_input, response)
                print(f"Bot: {response}")
        else:
            print("Invalid OTP.")
    else:
        print("Failed to send OTP.")
