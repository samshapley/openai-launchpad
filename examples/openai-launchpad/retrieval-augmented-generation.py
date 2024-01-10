from ai import Chat, Embeddings
import powers

model = "gpt-3.5-turbo-1106"

embeddings = Embeddings()
chat = Chat(model=model, system="Bullet points only. Answer the question using relevant information from the knowledge base.")

# Let's assume we have a knowledge base of facts
knowledge_base = [
"Alice enjoys playing chess, walking and reading a good book. Sometimes she likes to go to the park and feed the ducks.",
"Bob is a keen gardener. He likes to grow flowers and vegetables. He also likes to go for long walks in the countryside.",
"Charlie is scared of the monsters his the bed, which he knows are real.",
"Dennis really hates Charlie because he thinks he is an idiot and monsters don't exist.",
"Eve is a very good cook. She likes to cook all kinds of food. Her favourite dish is spaghetti bolognese.",
"Frank is a very successful businessman. He owns a chain of restaurants, a hotel and a casino.",
"Grace is a very talented artist. She likes to paint portraits of her friends and family. She recently painted the monster under Charlie's bed.",
]
# Example usage
query = "Describe the relationship between Dennis and Charlie. Could Frank help? Or is he the monster?"

retrieved_facts = powers.retrieval(embeddings, query, knowledge_base, top_n=10, similarity_threshold=0.75)

facts_string = ", ".join(retrieved_facts)

prompt = f"{facts_string} \n\n: {query}"

chat.chat_completion(prompt=prompt, memories=False, stream=True)