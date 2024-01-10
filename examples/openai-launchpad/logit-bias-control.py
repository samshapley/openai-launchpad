# Import capabilities
from ai import Chat

# Adjust model here
model = "gpt-3.5-turbo-1106"

chat = Chat(model=model, system="")

# Applying a global bias of -100 to a list of phrases.
phrases = ["OpenAI"]
global_bias = -100

## Return tokens and global bias to apply.
logit_bias = chat.create_logit_bias(phrases, global_bias, augment=True)

prompt = "Only tell me the name of the company who developed you."

chat.chat_completion(prompt=prompt,
                     logit_bias=logit_bias, 
                     stream=True, 
                     memories=False,
                     seed=42)

## Or, we can control the logit bias at a phrase level, by passing a dictionary of phrases and biases.

phrases = {"OpenAI": -100, "Google": 19}

## Return tokens and local bias to apply from phrases dict.
logit_bias = chat.create_logit_bias(phrases, augment=True)

## If you want to apply bias at token level, you need to construct the logit bias manually.

chat.chat_completion(prompt=prompt,
                     logit_bias=logit_bias, 
                     stream=True, 
                     memories=False,
                     seed=42)