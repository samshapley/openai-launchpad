import json
from ai import Embeddings

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

def use_tools(messages, available_functions):

    """
    Executes the functions specified in the tool_calls using the available functions provided and integrates the responses into the messages.

    Args:
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

            # Dynamically select and call the function based on function_name
            if function_name in available_functions:
                function = available_functions[function_name]
                function_response = function(**function_args)

                # Integrate the tool response into the messages
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

    return messages

def retrieval(embeddings, query, knowledge_base, top_n=1, similarity_threshold=0.0):
    """
    Retrieves the most relevant facts based on a query from a given knowledge base.

    Args:
        embeddings (Embeddings): An instance of the Embeddings class.
        query (str): The query string to retrieve information for.
        knowledge_base (List[str]): A list of facts as strings.
        top_n (int): The number of top relevant facts to return.
        similarity_threshold (float): The minimum similarity score for a fact to be considered relevant.

    Returns:
        List[str]: A list of the most relevant facts above the similarity threshold.
    """

    # Create embeddings for the query
    query_embedding = embeddings.create_embeddings([query])[0]['embedding']
    
    # Create embeddings for all facts in the knowledge base
    facts_embeddings = embeddings.create_embeddings(knowledge_base)

    # Calculate similarity scores between the query and each fact
    similarity_scores = []
    for fact, fact_data in zip(knowledge_base, facts_embeddings):
        fact_embedding = fact_data['embedding']
        similarity = embeddings.cosine_similarity(query_embedding, fact_embedding)
        if similarity >= similarity_threshold:
            similarity_scores.append((fact, similarity))
    
    # Sort the facts by similarity score in descending order
    sorted_facts = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Return the top_n most relevant facts above the threshold
    top_facts = [fact for fact, _ in sorted_facts[:top_n]]

    return top_facts if top_facts else None