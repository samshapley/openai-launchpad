import json

def extract_tool_calls(messages):
    """
    Extracts tool calls from the last message in the messages list.

    :param messages: List of message objects from the chat completion.
    :return: List of tool calls extracted from the last message.
    """
    last_message = messages[-1]

    try:
        tool_calls = last_message["tool_calls"]
    except KeyError:
        tool_calls = None
        
    return tool_calls

def use_tools(messages, available_functions):
    """
    Executes the functions specified in the tool_calls using the available functions provided and integrates the responses into the messages.

    :param messages: List of message objects from the chat completion.
    :param available_functions: Dictionary of available functions to be used.
    :return: The updated list of messages with the tool responses integrated.
    """
    tool_calls = extract_tool_calls(messages)

    if tool_calls is not None:
        for tool_call in tool_calls:
            function_name = tool_call["function"]["name"]
            function_args = json.loads(tool_call["function"]["arguments"])

            # Dynamically select and call the function based on function_name
            if function_name in available_functions:
                function = available_functions[function_name]
                function_response = function(**function_args)

                # Integrate the tool response into the messages
                function_response_json = json.dumps(function_response)
                messages.append({
                    "tool_call_id": tool_call["id"],
                    "role": "tool",
                    "name": function_name,
                    "content": function_response_json,
                })
            else:
                print(f"Function {function_name} not found in available_functions.")
        else:
            pass

    return messages