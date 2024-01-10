
# Import the abilities
from ai import Chat, Images, Vision, Embeddings

# Initialize the classes
chat = Chat(model="gpt-4", system = "You create a prompt for an image to be generated about the given topic: \n\n")
image = Images()
vision = Vision(system="Imagine you are reverse engineering the prompt used to generate this image. Give the prompt and nothing else.")
embedding = Embeddings()

# Use the Chat class to generate a description of the scene
completion = chat.chat_completion("The Lost City of Atlantis", memories=False, stream=True)

# Extract the description from the completion object
description = completion["response"]

# Use the Image class to generate an image based on the description
image_generation = image.generate_image(description, display_image=True, save_image=True)

# Use the Vision class to describe the generated image
vision_completion = vision.vision_completion("", image_paths=[image_generation["path_to_image"]], memories=False, stream=True)

# Calculate the cosine similarity between the original description and the image description
similarity = embedding.string_similarity(description, vision_completion["response"])

# Print the results
print(f"Original description: {description}")
print(f"Image description: {vision_completion['response']}")
print(f"Cosine similarity: {similarity}")