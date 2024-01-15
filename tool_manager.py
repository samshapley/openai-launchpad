import json

def extract_tool_calls(messages):
    """
    Extracts tool calls from the last message in the messages list.

    Args:
        messages (List[dict]): List of message objects from the chat completion.

    Returns:
        List[dict]: List of tool calls extracted from the last message, or None if no tool calls are present.
    """
    last_message = messages[-1]

    try:
        tool_calls = last_message["tool_calls"]
    except KeyError:
        tool_calls = None
        
    return tool_calls

def use_tools(chat_instance, messages, available_functions):

    """
    Executes the functions specified in the tool_calls using the available functions provided and integrates the responses into the messages.

    Args:
        chat_instance (Chat): An instance of the Chat class.
        messages (List[dict]): List of message objects from the chat completion.
        available_functions (dict): Dictionary of available functions to be used, with function names as keys and callable functions as values.

    Returns:
        List[dict]: The updated list of messages with the tool responses integrated.
    """
    tool_calls = extract_tool_calls(messages)

    if tool_calls is not None:
        for tool_call in tool_calls:
            function_name = tool_call["function"]["name"]
            function_args = json.loads(tool_call["function"]["arguments"])

            if function_name in available_functions:
                function = available_functions[function_name]
                function_response = function(**function_args)

                function_response_json = json.dumps(function_response)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": function_name,
                    "content": function_response_json,
                })
            else:
                print(f"Function {function_name} not found in available_functions.")
    else:
        pass

    chat_instance.messages = messages

    return messages

def get_tool_functions(tool_calls):
    """
    Extracts function names and arguments from a list of tool calls.

    Args:
        tool_calls (List[dict]): List of tool call objects.

    Returns:
        List[dict]: List of dictionaries with function names and arguments.
    """
    functions = []
    for tool_call in tool_calls:
        function_info = {
            "name": tool_call["function"]["name"],
            "arguments": json.loads(tool_call["function"]["arguments"])
        }
        functions.append(function_info)
    return functions