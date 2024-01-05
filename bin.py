def log_all_spans(span, name):
    # Log the current span
    span.log(name=name)

    for attribute, value in vars(span._span).items():
        if attribute == "child_spans":
            print(f"{attribute}: {len(value)}")
            for child_span in value:
                print(f"    {child_span}")

    # Access the child_spans attribute of the span object, if it exists
    child_spans = getattr(span._span, 'child_spans', [])
    
    # Check if child_spans is not empty
    if child_spans:
        # Iterate over each child span
        for child_span in child_spans:
            # Construct the Trace object with _model_dict set to None and _span set to the child_span object
            trace = {"trace": trace_tree.WBTraceTree(child_span)}
            trace.log(name=name)

## figure out how to construct the trace object for the nested spans
## figure out how to log the trace object
## we want a single log_all_spans function for the end of an application run to log everything to wandb.
## could handle the multiagent case by having a method which updates the end time of the child spans in the chain leading to the current span
## is it possible to have it such that the user only has to include the parent span, and all the orchestration is automatically handled? Perhaps using metadata?

## Adding images


audio = Audio()
chat = Chat(model=model, system="Helpful assistant.")
wb = SpanManager(name="system")

wandb.init(project="openai-launchpad") ## Initialize wandb project

chain_name = "conversation"

agent_name = "agent"

agent_span = wb.wandb_span(span_kind = "agent", span_name=agent_name, parent_span=None)

chain_span = wb.wandb_span(span_kind = "chain", span_name=chain_name, parent_span=agent_span)

for i in range(3):

    # transcript = audio.record_and_transcribe()
    transcript = "This is a test."
    time.sleep(1)

    transcript_span = wb.wandb_span(
        span_kind="tool",
        span_name="transcription",
        parent_span=chain_span,
        inputs={"audio": "audio"},
        outputs={"transcript": transcript},
    )

    if "quit" in transcript:
        break

    completion, messages = chat.chat_completion(transcript, speak=False, stream=True, memories=True)

    # Log the LLM used.
    llm_span = wb.wandb_span(
        span_kind="llm",
        span_name="chat",
        parent_span = chain_span,
        inputs={"transcript": transcript},
        outputs={"completion": completion, "messages": messages},
        metadata={"model": model},
    )

    voice = "echo"
    # audio.speak(completion, voice=voice)
    time.sleep(2)

    # Log the speech span.
    speech_span, llm_span , agent_span = wb.wandb_span(
        span_kind="tool",
        span_name="speech",
        parent_span=llm_span,
        inputs={"completion": completion},
        outputs={"tool": "Completion spoken."},
        metadata={"voice": voice}
    )

    # update the chain span end time
    chain_span.end_time_ms = speech_span.end_time_ms

conversation_end_time_ms = round(datetime.now().timestamp() * 1000)
chain_span.end_time_ms = conversation_end_time_ms

# log_all_spans(chain_span, chain_name)

# # Log the chain span.
# chain_span.log(name=agent_name)

# Log the agent span.
agent_span.log(name=agent_name)

# End wandb run
wandb.finish()

# Initialize wandb project
wandb.init(project="openai-launchpad")

# Define the chain name for logging
chain_name = "image_generation_and_vision"

image_prompt = "The power of time in the palm of your hand."

# Start the chain for logging
chain_span, _ , _ = wb.wandb_span(span_kind="chain", span_name=chain_name, parent_span=None, root_span=None)

# Generate an image
images = Images()
response, path = images.generate_image(image_prompt, display_image=True, save_image=True)

# Log the image generation span
image_gen_span, _ , chain_span = wb.wandb_span(
    span_kind="tool",
    span_name="image_generation",
    inputs={"prompt": image_prompt},
    outputs={"response": response, "image_path": path},
    parent_span=None,
    root_span=chain_span,
    metadata={"display_image": True, "save_image": True}
)

# Vision system analysis
vision = Vision(system="You can only respond with bullet points.")
response, messages = vision.vision_completion(prompt="What is this image showing?", image_paths=[path], memories=True, stream=True)

# Log the vision system span
vision_span, _ , chain_span = wb.wandb_span(
    span_kind="llm",
    span_name="vision_description",
    inputs={"prompt": "What is this image showing?", "image_paths": [path]},
    outputs={"response": response, "messages": messages},
    parent_span=None,
    root_span=chain_span
)

chain_span.log(name=chain_name)

# End wandb run
wandb.finish()




def log_all_spans(span, name):
    # Log the current span
    span.log(name=name)

    for attribute, value in vars(span._span).items():
        if attribute == "child_spans":
            print(f"{attribute}: {len(value)}")
            for child_span in value:
                print(f"    {child_span}")

    # Access the child_spans attribute of the span object, if it exists
    child_spans = getattr(span._span, 'child_spans', [])
    
    # Check if child_spans is not empty
    if child_spans:
        # Iterate over each child span
        for child_span in child_spans:
            # Construct the Trace object with _model_dict set to None and _span set to the child_span object
            trace = {"trace": trace_tree.WBTraceTree(child_span)}
            trace.log(name=name)
