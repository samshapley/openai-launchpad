from ai import Chat

model = "gpt-3.5-turbo-1106"

import powers

chat = Chat(model=model, system="Helpful robot.")

# Dummy function to simulate an API call to get weather data
def get_current_weather(location, unit="celsius"):
    # In a real scenario, this function would make an API call to a weather service
    weather_data = {
        "Tokyo": {"temperature": "-100", "unit": unit},
        "San Francisco": {"temperature": "30", "unit": unit},
        "Paris": {"temperature": "40", "unit": unit}
    }
    return weather_data.get(location, {"temperature": "unknown"})

# Dictionary mapping function names to actual function objects
available_functions = {
    "get_current_weather": get_current_weather,
    # Add other functions here as needed
}

# Define the tool that the model can use
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city, e.g. San Francisco",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]

# Get the initial response from the model
completion = chat.chat_completion(
    prompt="What's the temperature and weather in Paris vs Tokyo as a rhyming couplet?",
    tools=tools,
    tool_choice="auto",
    return_tool_calls=True,
    return_messages=True,
    stream=False,
)

# Get the messages from the completion object
messages = completion["messages"]

## Use tools is a power that applies any tools on the last message, and returns the updated messages with the responses from the tools integrated.   
chat.use_tools(available_functions)

# Get the final response from the model after the tool has been used
final_completion = chat.chat_completion(
    prompt="",
    stream=True,
)