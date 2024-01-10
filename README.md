# OpenAI Launchpad

![A retro pixel art image of a robot on a launchpad trampoline, similar to the first design, but with a unique twist_ the robot has a visible neural net-2](https://github.com/samshapley/openai-launchpad/assets/93387313/df904c4a-d2b9-4d25-99c8-5282ad01f760)

### What is OpenAI Launchpad?

`this is not affiliated with OpenAI in any way`

OpenAI Launchpad is a Python library that makes it easy to use the OpenAI API. It's designed to be easy to use, and easy to understand.

The different "abilities" OpenAI offer are all made readily available in a single file, **ai.py**, which you can import and use in your own projects. 

Use this to quickly prototype ideas, or to create entire intelligent systems.

This is a layer to enhance the OpenAI API. If you need to avoid open-source, everyone knows these LLMs are the good stuff.

### Installation

```python
git clone git@github.com:samshapley/openai-launchpad.git

cd openai-launchpad

pip install -r requirements.txt
```

### Integrate with your own projects

Use **ai.py** as an existing backend for hackathon systems or projects that make use of the OpenAI API.

### What can it do?

Each endpoint [in the API](https://platform.openai.com/docs/api-reference) has its own class, and every feature offered by that endpoint is built into that class already.

- **Chat**: Engage in conversations with AI, with options to remember context, stream and speak responses, and apply logit biases to control the AI's output.
- **Vision**: Process images using AI, including encoding images to base64, constructing image messages, and generating vision completions.
- **Audio**: Generate and manipulate audio, including text-to-speech, transcription, translation, and recording from the microphone.
- **Images**: Generate images from text prompts with the ability to display, save, and describe the generated images.
- **Embeddings**: Calculate string similarity using embeddings, providing a measure of semantic similarity between texts.
- **FineTuner**: Fine-tune models with custom datasets, manage fine-tuning jobs, and utilize fine-tuned models for completions.

**powers.py** is a file that contains all the things the AIs can do , i.e external tools.

### Integration with Weights and Biases

See the `examples/wandb/README.md` for full details. 

Launchpad can be easily combined with Wandb for easy tracking of new AI systems.

### Usage

See the `examples.ipynb` file for a full example of how to use the library.

Here are some other simple examples:

Sure, here are some basic examples of how to use the ai.py script:

1. Chat Completion

```python
from ai import Chat
chat = Chat(model="gpt-3.5-turbo", system="Helpful assistant.")
completion, messages = chat.chat_completion(prompt="Tell me a joke.")
```
Everything the chat endpoint accepts is available in `chat_completion`.

2. Vision Completion

```python
from ai import Vision
vision = Vision()
response, messages = vision.vision_completion(prompt="What is in this image?", image_paths=["path_to_your_image.jpg"])
```

3. Image Generation

```python
from ai import Images
images = Images()
response, path = images.generate_image("A chicken in a suit of armour", display_image=True, save_image=True)
```

4. Embeddings for String Similarity

```python
from ai import Embeddings
embedding = Embeddings()
similarity = embedding.string_similarity("string1", "string2")
```

5. Logit Bias

```python
from ai import Chat
chat = Chat(model="gpt-3.5-turbo", system="")
phrases = ["OpenAI"]
global_bias = -100
logit_bias = chat.create_logit_bias(phrases, global_bias, augment=True)
response, messages = chat.chat_completion(prompt="Only tell me the name of the company who developed you.", logit_bias=logit_bias)
```

6. Text to Speech

```python
from ai import Audio
audio = Audio()
audio.speak(text="The quick brown fox jumps over the lazy dog.", voice='echo')
```

7. Fine-Tuning a Model

```python
from ai import FineTuner
fine_tuner = FineTuner()
fine_tuning_job = fine_tuner.finetune_model(file_path='finetune-data.jsonl', batch_size='12', learning_rate_multiplier='0.0001', model_name='gpt-3.5-turbo', suffix='example', n_epochs=10)
```
