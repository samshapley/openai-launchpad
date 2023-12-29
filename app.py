import streamlit as st
from ai import Chat, Vision, Audio, Images

# Initialize the components
chat = Chat()
vision = Vision()
audio = Audio()
image = Images()

# Create a sidebar for navigation
st.sidebar.title("AI Services")
service = st.sidebar.radio("Choose a service:", ("Chat", "Vision", "Audio", "Cooking Assistant", "DALLE"))

# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

if service == "Chat":
    st.title("Chat with AI")

    # Display the chat history
    for message in st.session_state['chat_history']:
        st.text_area("", value=message, height=70, disabled=True)

    # Input for user message
    user_input = st.text_input("Enter your message:")

    if st.button("Send"):
        # Update the chat history with the user's message
        st.session_state['chat_history'].append("You: " + user_input)

        # Get the AI's response
        response, _ = chat.chat_completion(prompt=user_input)

        # Update the chat history with the AI's response
        st.session_state['chat_history'].append("AI: " + response)

        # Clear the input box after sending the message
        st.experimental_rerun()

elif service == "Vision":
    st.title("Image Analysis")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Save the uploaded image to a path and pass it to vision_completion
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.getvalue())
        response, _ = vision.vision_completion(prompt="Describe this image", image_paths=["temp_image.jpg"])
        st.text_area("Analysis:", value=response, height=200)

elif service == "Audio":
    st.title("Text to Speech")
    text_to_speak = st.text_input("Enter text to speak:")
    if st.button("Speak"):
        audio.speak(text=text_to_speak)
        st.success("Audio played successfully")

# Add a new elif block for the Image Generator service
elif service == "DALLE":
    st.title("Generate Images with AI")
    prompt = st.text_input("Enter a prompt for the image:")
    if st.button("Generate"):
        images = Images()
        response, image_path = images.generate_image(prompt=prompt, display_image=False, save_image=True)
        st.image(image_path, caption="Generated Image")

elif service == "Cooking Assistant":
    st.title("Virtual Cooking Assistant")
    uploaded_file = st.file_uploader("Upload an image of your ingredients", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Save the uploaded image to a path and pass it to vision_completion
        with open("ingredients_image.jpg", "wb") as f:
            f.write(uploaded_file.getvalue())
        response, _ = vision.vision_completion(prompt="Identify these ingredients.", image_paths=["ingredients_image.jpg"])
        st.text_area("Identified Ingredients:", value=response, height=200)
        recipe_prompt = f"Create a recipe using these ingredients: {response}"
        recipe_response, _ = chat.chat_completion(prompt=recipe_prompt)
        st.text_area("Suggested Recipe:", value=recipe_response, height=400)

# Run the Streamlit app
# To run the app, save this script as app.py and run `streamlit run app.py` in your terminal.