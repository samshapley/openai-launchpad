from ai import Chat, Images
import tool_manager as tm
import powers
import wandb
from wandb_logging import WandbSpanManager

# Define your functions here
def generate_paragraph_with_image(paragraph, image_prompt):
    return {
        "paragraph_added": True,
    }

# Dictionary mapping function names to actual function objects
available_functions = {
    "generate_paragraph_with_image": generate_paragraph_with_image,
   
}

# Define the tools that the model can use
tools = [
    {
        "type": "function",
        "function": {
            "name": "generate_paragraph_with_image",
            "description": "Generate a paragraph with an associated image for a storybook",
            "parameters": {
                "type": "object",
                "properties": {
                    "paragraph": {
                        "type": "string",
                        "description": "Generate the paragraph for a storybook.",
                    },
                    "image_prompt": {
                        "type": "string",
                        "description": "Generate the image prompt for a storybook, which accompanies the paragraph.",
                    },
                },
                "required": ["paragraph, image_prompt"],
            },
        },
    },
]

system = """
You are a storybook writer. You will call the functions to iterate on the existing story.
Prompts should be written in such a way that the story is deep and interesting.
Image pormpts should describe the specific output of what the image should look like. The system uses a highly advanced image generation model to generate the image, and with
very clear instructions, styles, postions and descriptions images can be replciated and closely controlled. 
We want our stories to maintain cohesion and continuity, so please make sure that the story is consistent and makes sense.
Visually, the prompts should therefore maintain a consistent style and theme. A character should never be introduced without a detailed description of their appearance, age and clothing. Sections of previous prompts can be used to maintain visual continuity.
Remember, the image generator is stateless, so you will need to re-describe exactly the same way aspects of image content that you want to maintain.
"""

# Initialize wandb project
wandb.init(project="storybook-generator")

# Create a WandbSpanManager instance
wb = WandbSpanManager(name="storybook")

# Initialize Chat and Images
chat = Chat(model="gpt-4-1106-preview", system=system)
images = Images()

storybook = []

# Start the chain for logging
chain_span = wb.wandb_span(span_kind="chain", span_name="storybook_creation", parent_span_id=None)

# Table to log the storybook data
columns = ["paragraph", "image_path"]
storybook_table = wandb.Table(columns=columns)

while True:
    # Prompt if its the first paragraph
    if len(storybook) == 0:
        prompt = input("What is the storybook about? ")
    else:
        prompt = "Continue the next paragraph."

    # Generate a story paragraph
    paragraph_completion = chat.chat_completion(prompt=prompt, stream=True, memories=True, return_messages = True, tools=tools, tool_choice="auto", return_tool_calls=True)
    # Get the paragraph from the completion object
    paragraph = paragraph_completion["response"]

    # Log the paragraph generation span
    paragraph_span = wb.wandb_span(
        span_kind="tool",
        span_name="paragraph_generation",
        inputs={"prompt": prompt},
        outputs=paragraph_completion,
        parent_span_id=chain_span
    )

    tool_calls = tm.get_tool_functions(paragraph_completion["tool_calls"])

    messages = chat.use_tools(available_functions)

    for next_section in tool_calls:
        if next_section["name"] == "generate_paragraph_with_image":
            paragraph = next_section["arguments"]["paragraph"]
            image_prompt = next_section["arguments"]["image_prompt"]

            print(paragraph)

            # Generate an image to describe the scene
            image_completion = images.generate_image(prompt=image_prompt, display_image=True, save_image=True, fix_prompt=True)
            image_path = image_completion["path_to_image"]

            # Log the image generation span
            image_span = wb.wandb_span(
                span_kind="tool",
                span_name="image_generation",
                inputs={"prompt": paragraph},
                outputs={"image_path": image_path},
                parent_span_id=chain_span
            )


            # Add paragraph and image path to the storybook
            storybook.append((paragraph, image_path))

            # Add data to the table
            storybook_table.add_data(paragraph, wandb.Image(image_path))
        
    # Check if the user wants to continue generating the story
    continue_story = input("Do you want to add another paragraph to the story? (y/n): ")
    if continue_story.lower() != 'y':
        break

# Log the table
wandb.log({"Storybook": storybook_table})

wb.log_top_level_span()

wandb.finish()

print("The storybook is complete!")