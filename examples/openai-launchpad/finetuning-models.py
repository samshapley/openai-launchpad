from ai import FineTuner

# Create an instance of the FineTuner class
fine_tuner = FineTuner()
 
# Define the path to your JSONL file
jsonl_file_path = 'assets/finetune-data.jsonl'

# Upload the file and start the fine-tuning process
fine_tuning_job = fine_tuner.finetune_model(
    file_path=jsonl_file_path,
    batch_size='12',
    learning_rate_multiplier='0.0001',
    model_name='gpt-3.5-turbo',
    suffix='example',
    n_epochs=10,  # for example

)

# Print the fine-tuning job details
print(fine_tuning_job)

## Once it is complete, you can use the fine-tuned model like this:

# finetuned_job = fine_tuner.retrieve_finetuning_job(fine_tuning_job.id)

# finetuned_job.fine_tuned_model

# chat = Chat(model=finetuned_job.fine_tuned_model, system="")

# completion = chat.chat_completion(prompt="What is the meaning of life?", memories=False, stream=True)