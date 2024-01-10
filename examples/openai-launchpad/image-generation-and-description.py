# Import capabilities
from ai import Vision, Images

# Adjust model here
model = "gpt-3.5-turbo-1106"

images = Images()

image_generation = images.generate_image("What the universe looks like to an outside observer.", fix_prompt=False, response_format="url", display_image=True, save_image=True)

vision = Vision(system="You can only respond with bullet points.")

vision.vision_completion(prompt="What is this image showing?", image_urls=[image_generation["url"]], memories=False, stream=True)