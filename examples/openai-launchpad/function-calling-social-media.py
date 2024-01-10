from ai import Chat
import powers

model = "gpt-3.5-turbo-1106"

chat = Chat(model=model, system="Resourceful assistant.")

# Dummy function to simulate checking a user's social media mentions
def check_social_mentions(username):
    # In a real scenario, this function would interact with a social media API
    mention_data = {
        "alice": ["Just had a great experience with @alice's bakery!", "@alice's new cake design looks amazing!"],
        "bob": ["@bob's tech reviews are always so insightful.", "Can't wait for @bob's next podcast episode!"],
        "carol": ["@carol's workout tips have really helped me improve my routine.", "So inspired by @carol's health journey!"]
    }
    return mention_data.get(username, ["No mentions found."])

# Dictionary mapping function names to actual function objects
available_functions = {
    "check_social_mentions": check_social_mentions,
    # Add other functions here as needed
}

# Define the tool that the model can use
tools = [
    {
        "type": "function",
        "function": {
            "name": "check_social_mentions",
            "description": "Check the latest social media mentions for a user",
            "parameters": {
                "type": "object",
                "properties": {
                    "username": {
                        "type": "string",
                        "description": "The social media username to check",
                    },
                },
                "required": ["username"],
            },
        },
    }
]

# Get the initial response from the model
completion = chat.chat_completion(
    prompt="What are people saying about bob on social media?",
    tools=tools,
    tool_choice="auto",
    stream=False,
    return_messages=True,
)

# Get the messages from the completion object
messages = completion["messages"]


messages = powers.use_tools(messages, available_functions)

# Get the final response from the model after the tool has been used
completion = chat.chat_completion(
    prompt="reply as a sea shanty",
    messages=messages,
    stream=True,
)