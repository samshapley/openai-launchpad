# set directory so i can import packages from one level up
import time
import sys
import os
import wandb
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai import Chat, Audio
from wandb_logging import WandbSpanManager

# Adjust model globally here or modify in function calls. Vision model cannot be changed, for now ;) 
model = "gpt-3.5-turbo-1106"

wb = WandbSpanManager(name="system")

wandb.init(project="openai-launchpad") ## Initialize wandb project

system_prompt = "You're a helpful assistant. You help people with their problems. You're a good listener."

audio = Audio()
chat = Chat(model=model, system=system_prompt)

chain_name = "conversation"

agent_name = "agent"

agent_span = wb.wandb_span(span_kind = "agent", span_name=agent_name, parent_span_id=None)

while True:

    chain_span = wb.wandb_span(span_kind = "chain", span_name=chain_name, parent_span_id=agent_span)

    transcript = audio.record_and_transcribe()
    # transcript = "This is a test."
    time.sleep(1)

    transcript_span = wb.wandb_span(
        span_kind="tool",
        span_name="transcription",
        parent_span_id=chain_span,
        inputs={"audio": "audio"},
        outputs={"transcript": transcript},
    )

    if "quit" in transcript:
        break

    completion = chat.chat_completion(transcript, speak=False, stream=True, memories=True)

    # Log the LLM used.
    llm_span = wb.wandb_span(
        span_kind="llm",
        span_name="chat",
        parent_span_id = chain_span,
        inputs={"system_prompt": system_prompt, "transcript": transcript},
        outputs=completion,
        metadata={"model_name": model}
    )

    voice = "echo"
    audio.speak(completion["response"], voice=voice, speed=1.5)

    # Log the speech span.
    speech_span = wb.wandb_span(
        span_kind="tool",
        span_name="speech",
        parent_span_id = chain_span,
        inputs={"text": completion["response"]},
        outputs={"tool": "Completion spoken."},
        metadata={"voice": voice}
    )


wb.update_span_by_id(agent_span, inputs={"transcript": transcript}, outputs=completion)

wb.log_top_level_span()

# End wandb run
wandb.finish()