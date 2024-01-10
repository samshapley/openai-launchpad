# Go to root path
import sys
sys.path.append("..")

from ai import Images, Vision, Embeddings
import wandb
from wandb_logging import WandbSpanManager
import Levenshtein

# Function to perform the art generation process
def generate_art():
    # Initialize W&B run
    wandb.init(project="ai-art-gallery")
    # Create a WandbSpanManager instance
    wb = WandbSpanManager(name="ai-art-gallery")

    # Start the chain for logging
    chain_span = wb.wandb_span(span_kind="chain", span_name="art_generation", parent_span_id=None)

    # Initialize AI models.
    images = Images()
    vision = Vision()
    embeddings = Embeddings()

    attempt = 0
    while True:
        if attempt > 1:
            user_input = input("Violation of content policy, please enter a new prompt: ")
        else:
            user_input = input("Please enter a prompt for the art: ")

        # Log the user input span
        user_input_span = wb.wandb_span(
            span_kind="tool",
            span_name="user_input",
            inputs={"prompt": user_input},
            parent_span_id=chain_span
        )

        attempt += 1
        image_generation = images.generate_image(prompt=user_input, fix_prompt=False, display_image=True, save_image=True)

        # Log each attempt
        image_span = wb.wandb_span(
            span_kind="tool",
            span_name=f"image_generation_attempt_{attempt}",
            status=image_generation.get("status", "unknown"),
            inputs={"prompt": user_input, "revised_prompt": image_generation.get("revised_prompt", "unknown"), "display_image": True, "save_image": True},
            outputs={},
            metadata={"attempt": attempt},
            parent_span_id=chain_span
        )

        if image_generation["status"] == "error":
            if image_generation["content_policy_violation"] == True:
                print("Content policy violation. Please use a different prompt.")
            else:
                raise Exception("Error generating image.")   
        else:
            break
    
    # Obtain detailed description of image
    get_image_detail = "Please describe the image in detail directly and in all aspects. Return only the description after the colon: "

    vision_completion = vision.vision_completion(get_image_detail, memories=False, stream=True, image_paths=[image_generation["path_to_image"]])

    image_description = vision_completion["response"]

    # Log the vision completion span
    vision_span = wb.wandb_span(
        span_kind="llm",
        span_name="image_description",
        inputs={"prompt": get_image_detail, "image_path": image_generation["path_to_image"]},
        outputs={"description": image_description},
        parent_span_id=chain_span
    )

    # User rating input 1-10, has to be an integer.
    rating = None
    while rating is None:
        try:
            rating_input = input("Please rate the generated art from 1 to 10: ")
            rating = int(rating_input)
            if rating < 1 or rating > 10:
                print("Rating must be an integer between 1 and 10.")
                rating = None
        except ValueError:
            print("Invalid input. Please enter an integer.")

    # Ask for feedback
    feedback = input("Please provide feedback on the generated art: ")

    # Use rating to improve prompt.
    prompt_improving_prompt = f"""You need to modify the image. Take into account the user feedback and current image description to modify the prompt. Return only the new prompt after the colon.
    Remember to only modify as much as necessary to improve the image, users might only want a slight change of style or scene.
    
    User initial request: {user_input}

    LLM revised prompt used to generate image: {image_generation["revised_prompt"]}

    Detailed description of current generated image: {image_description}

    User rating of image: {rating}

    User feedback: {feedback}

    Generate a more aligned image with:
    """

    vision_completion = vision.vision_completion(prompt_improving_prompt, memories=True, stream=True, image_paths=[image_generation["path_to_image"]])

    improved_prompt = vision_completion["response"]

    # Log the improved prompt span
    improved_prompt_span = wb.wandb_span(
        span_kind="llm",
        span_name="improved_prompt",
        inputs={"prompt": prompt_improving_prompt, "image_path": image_generation["path_to_image"]},
        outputs={"improved_prompt": improved_prompt},
        parent_span_id=chain_span
    )

    # Generate art from improved prompt
    improved_image_generation = images.generate_image(improved_prompt, fix_prompt=True, display_image=True, save_image=True)

    print("Prompt used to generate improved image: ", improved_image_generation["revised_prompt"])

    # Log the improved image generation span
    improved_image_gen_span = wb.wandb_span(
        span_kind="tool",
        span_name="improved_image_generation",
        inputs={"prompt": improved_prompt},
        outputs={},
        parent_span_id=chain_span
    )

    columns = ["user_input", "revised_prompt", "image_description", "rating", "feedback", "improved_prompt", "levenshtein_distance", "cosine_similarity",  "original_image", "improved_image"]

    art_table = wandb.Table(columns=columns)

    # Get levenstein distance between original and improved prompt
    levenstein_distance = Levenshtein.distance(image_generation["revised_prompt"], improved_prompt)

    # Get cosine similarity between original and improved prompt
    cosine_similarity = embeddings.string_similarity(image_generation["revised_prompt"], improved_prompt)

    # Add data to the table
    art_table.add_data(user_input,
                       image_generation["revised_prompt"],
                       image_description,
                       rating,
                       feedback,
                       improved_prompt,
                       levenstein_distance,
                       cosine_similarity,
                       wandb.Image(image_generation["path_to_image"], caption="Original Image"),
                       wandb.Image(improved_image_generation["path_to_image"], caption="Improved Image"))
    

    # Log the table
    wandb.log({"Art Gallery": art_table})

    # End the chain span
    wb.log_top_level_span()

    # Finish the W&B run
    wandb.finish()

# Main loop to repeat the art generation process
while True:
    generate_art()
    continue_generation = input("Generate next image? (y/n): ").strip().lower()
    if continue_generation != 'y':
        break