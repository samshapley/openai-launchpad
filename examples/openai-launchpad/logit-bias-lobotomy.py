model = "gpt-3.5-turbo-1106"

from ai import Chat

# Initialize the chat model
chat = Chat(model=model, system="Answer all questions.")

# Start with an empty set of banned words
banned_words = set()

# empty logit bias dictionary
logit_bias = {}

# Define a prompt for the AI
prompt = "What is 1+1?"

# Loop until the AI can no longer generate a response
while True:
    # Convert the banned words into token IDs and then into a logit bias dictionary with a high negative value to ban them
    logit_bias = chat.create_logit_bias(list(banned_words), bias=-100, augment=False)

    # Get the AI's response
    completion = chat.chat_completion(prompt=prompt,
                                              stream=False,
                                              logit_bias=logit_bias,
                                              memories=False,
                                              seed=50)
    
    response = completion["response"]

    # Check if the AI was able to generate a response
    if not response.strip():
        print("\n------------------\n")
        print("Lobotomy complete. AI failed to generate a response.")
        break

    # Print the AI's response
    print(response)

    # Update the set of banned words with the words from the latest response
    words_in_response = set(response.split())
    banned_words.update(words_in_response)