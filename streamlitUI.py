import streamlit as st

# Function to simulate chatbot response
def chatbot_response(user_input):
    # This is a placeholder for your chatbot logic
    responses = {
        "hi": "Hello! How can I assist you today?",
        "hello": "Hi there! How can I help?",
        "how are you?": "I'm just a bot, but I'm doing well! How can I help you?",
    }
    return responses.get(user_input.lower(), "I'm sorry, I didn't understand that.")

# Function to handle chat UI
def chat_ui():
    # Set page configuration
    st.set_page_config(page_title="Modern Chatbot", page_icon=":speech_balloon:", layout="wide")
    
    # Add custom CSS for background image, colors, and text box styling
    st.markdown(
        """
        <style>
        body {
            background-image: url("https://www.google.com/imgres?imgurl=https%3A%2F%2Ft3.ftcdn.net%2Fjpg%2F06%2F92%2F77%2F34%2F360_F_692773469_YXy6XzZ41UoNKLg4RJl3JMiupAXK3zJV.jpg&tbnid=DFWZ_xncVBPZZM&vet=10CAQQxiAoAmoXChMI-JqArrKNigMVAAAAAB0AAAAAEA8..i&imgrefurl=https%3A%2F%2Fstock.adobe.com%2Fsearch%3Fk%3Dpharmacy%2Bbackground&docid=oDCehWXMEyZJXM&w=643&h=360&itg=1&q=pharmacy%20images%20wallpaper&client=safari&ved=0CAQQxiAoAmoXChMI-JqArrKNigMVAAAAAB0AAAAAEA8");
            background-size: cover;
            background-position: center;
        }
        .stChatMessage {
            background-color: gray;
            color: white;
        }
        .stTitle {
            color: #4CAF50;
        }
        .stButton>button {
            background-color: gray;
            color: white;
        }
        /* Text box color change to light grey with opacity 0.8 */
        .stTextInput>div>div>input {
            background-color: rgba(211, 211, 211, 0.8); /* Light grey with opacity 0.8 */
            color: #000;  /* Black text color for contrast */
            border: 1px solid #4CAF50; /* Optional: Green border for the text box */
        }
        </style>
        """, unsafe_allow_html=True)
    
    # Title
    st.title("Chat with Me!")
    
    # Initialize chat history in session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        if message['role'] == 'user':
            st.chat_message(message['role']).markdown(f"**You:** {message['content']}")
        else:
            st.chat_message(message['role']).markdown(f"**Bot:** {message['content']}")

    # User input
    user_input = st.chat_input("Type a message...")

    if user_input:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Get bot response
        bot_response = chatbot_response(user_input)
        
        # Add bot message to history
        st.session_state.messages.append({"role": "bot", "content": bot_response})
        
        # Refresh the chat to show the new message
        st.rerun()

if __name__ == "__main__":
    chat_ui()
