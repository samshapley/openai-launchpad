#### Moving memory from a conversation LLM to the vision LLM.
from ai import Chat, Vision

model = "gpt-3.5-turbo-1106"

chat = Chat(model=model, system="")

vision = Vision()

completion_1 = chat.chat_completion(prompt="What is your purpose?", stream=False, memories=True, return_messages=True)

### This should respond with something like "You just asked, 'What is your purpose?' if the memory transfer worked."
completion_2 = vision.vision_completion(prompt="What did I just say and how did you respond?", messages=completion_1["messages"], stream = True, speak = False)

print(completion_2)