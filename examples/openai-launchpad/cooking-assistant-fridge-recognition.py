from ai import Chat, Vision, Images

vision = Vision(system="Identify ingredients in the image and make a list.")
images = Images()

# User uploads an image of ingredients they have
ingredient_image_path = "assets/fridge.jpg"

# Vision AI identifies the ingredients
vision_completion = vision.vision_completion(prompt="Identify these ingredients.", image_paths=[ingredient_image_path], stream=True)

# Extract the identified ingredients from the response
ingredients = vision_completion["response"]

chat = Chat(model="gpt-3.5-turbo", system="Generate a recipe based on the following ingredients.")
recipe_prompt = f"Create a recipe using these ingredients: {ingredients}"

# Generate a recipe
completion = chat.chat_completion(prompt=recipe_prompt, stream=True)
recipe = completion["response"]

# Generate an image of the dish
images.generate_image(prompt=recipe, display_image=True, save_image=True)