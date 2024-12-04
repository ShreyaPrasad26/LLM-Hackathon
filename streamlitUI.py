'''
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

##Second iteration

import streamlit as st
from app import PharmaKnowledgeAssistant  # Importing the backend class

# Initialize the backend assistant
assistant = PharmaKnowledgeAssistant()

# Streamlit App
st.title("Pharma Knowledge Assistant")
st.sidebar.title("Select Functionality")

# Sidebar for selecting functionalities
tabs = ["Chat", "Product Knowledge Base", "Medicine Recommender", "Alternative Medicines", "Summarizer"]
selected_tab = st.sidebar.radio("Choose a feature:", tabs)

if selected_tab == "Chat":
    st.header("Chat with Pharma Assistant")
    query = st.text_input("Enter your question:")
    if st.button("Get Answer"):
        response = assistant.agent.run(query)
        st.write("Response:", response)

elif selected_tab == "Product Knowledge Base":
    st.header("Search Product Knowledge Base")
    query = st.text_input("What would you like to know?")
    if st.button("Search"):
        response = assistant._query_product_database(query)
        st.write("Response:", response)

elif selected_tab == "Medicine Recommender":
    st.header("Get Medicine Recommendations")
    symptoms = st.text_area("Describe the symptoms:")
    if st.button("Recommend Medicines"):
        response = assistant._recommend_medicines(symptoms)
        st.write("Recommendations:", response)

elif selected_tab == "Alternative Medicines":
    st.header("Find Alternative Medicines")
    medicine_name = st.text_input("Enter the medicine name:")
    if st.button("Find Alternatives"):
        response = assistant._generate_alternatives(medicine_name)
        st.write("Alternatives:", response)

elif selected_tab == "Summarizer":
    st.header("Summarize Medicine Information")
    medicine_info = st.text_area("Enter detailed medicine information:")
    if st.button("Summarize"):
        response = assistant._summarize_medicine(medicine_info)
        st.write("Summary:", response)

# Third iteration

import streamlit as st
from app import PharmaKnowledgeAssistant  # Importing the backend assistant

# Initialize the assistant
assistant = PharmaKnowledgeAssistant()

# Streamlit UI
st.title("Pharma Knowledge Assistant")
st.subheader("Ask any pharmaceutical or medical-related query")

# User query input
query = st.text_input("Enter your query (type 'quit' to exit):", "")

# Display response
if st.button("Submit"):
    if query.lower() == "quit":
        st.warning("You have exited the Pharma Knowledge Assistant. Refresh the page to restart.")
    elif query.strip():
        # Call the process_query function from the backend
        response = assistant.process_query(query)
        st.success("Assistant Response:")
        st.write(response)
    else:
        st.error("Please enter a valid query!")

# Footer
st.write("---")
st.write("Type 'quit' to stop interacting with the assistant.")
'''
import streamlit as st
from app import PharmaKnowledgeAssistant  # Importing the backend assistant

# Initialize the assistant
assistant = PharmaKnowledgeAssistant()

# Streamlit UI
st.title("Welcome to Pharma Knowledge Assistant!")
st.sidebar.header("Features")
st.sidebar.markdown("Select a feature from the list below:")

# Sidebar options
features = {
    "Chat": "Ask any pharmaceutical or medical-related query",
    "Product Knowledge Base": "Search for product information",
    "Medicine Recommender": "Get recommendations based on symptoms",
    "Alternative Medicines": "Find substitutes for medicines",
    "Summarizer": "Summarize detailed medicine information",
}
selected_feature = st.sidebar.selectbox("Choose a feature:", list(features.keys()))

# Main section
st.header(features[selected_feature])

# Feature-specific inputs and processing
if selected_feature == "Chat":
    query = st.text_input("Enter your question:")
    if st.button("Submit"):
        if query.strip():
            response = assistant.agent.run(query)
            st.write("Response:", response)
        else:
            st.error("Please enter a valid question.")

elif selected_feature == "Product Knowledge Base":
    query = st.text_input("What would you like to know?")
    if st.button("Search"):
        if query.strip():
            response = assistant._query_product_database(query)
            st.write("Response:", response)
        else:
            st.error("Please enter a valid query.")

elif selected_feature == "Medicine Recommender":
    symptoms = st.text_area("Describe the symptoms:")
    if st.button("Recommend"):
        if symptoms.strip():
            response = assistant._recommend_medicines(symptoms)
            st.write("Recommendations:", response)
        else:
            st.error("Please provide symptoms for recommendations.")

elif selected_feature == "Alternative Medicines":
    medicine_name = st.text_input("Enter the medicine name:")
    if st.button("Find Alternatives"):
        if medicine_name.strip():
            response = assistant._generate_alternatives(medicine_name)
            st.write("Alternatives:", response)
        else:
            st.error("Please enter a valid medicine name.")

elif selected_feature == "Summarizer":
    medicine_info = st.text_area("Enter detailed medicine information:")
    if st.button("Summarize"):
        if medicine_info.strip():
            response = assistant._summarize_medicine(medicine_info)
            st.write("Summary:", response)
        else:
            st.error("Please provide valid medicine information to summarize.")

# Footer
st.write("---")
st.caption("Powered by Pharma Knowledge Assistant")
