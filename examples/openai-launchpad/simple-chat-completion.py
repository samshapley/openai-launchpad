from ai import Chat

model = "gpt-3.5-turbo-1106"

chat = Chat(model=model, system="Helpful robot")

object = "pizza"

# memories control whether the AI remembers the conversation or not, provided Chat is not reinitialized.
completion = chat.chat_completion(
    prompt=f"Write an article about {object}",
    memories=True,
    temperature=0.9, 
    presence_penalty=0.6,
    seed=42,
    return_logprobs=True,
    stream=True)