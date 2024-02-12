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