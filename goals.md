## To Do

check all docstrings are correct and up to date. (build a general ai check for this in the powers folder under "health_check.py") -- explore the idea further of AI systme having its own "powers". i,e adjusting its own parameters and code.

better image transitons between models, i.e return prompt, json, path to save.

streamlit or gradio on top of this.

better logging

add finetune lookup by suffix, or time range

logprobs with streaming

make it so tools can be activated within the chat completion call.

then build wandb logging into this.

local cost logging

better audio streaming, and calibration of mic.

update the audio docstringts and input/output to be clearer

https://cookbook.openai.com/examples/gpt_with_vision_for_video_understanding video understanding app.


## Self Control and Discovery of OpenAI Parameters

-- allow model to explore the use of unknown parameters to test what happens, and discover if it can reconstruct the code it is running on, and what each parameter does.
-- compare different models, and plot how well the code is reconstructed by this playing around with parameters, and seeing the effect on their own responses.

## Overall End Goal

Finetune on the examples ipynb file
--- this model should then be able to build apps on command:
--- expand the ipynb file
--- and keep the base framework
--- any changes and the example script should all still work and be evaluated
--- and examples modfied to match the new framework if they break
--- then once this loop is completed, we redo the finetuning
--- and see if the model is better at making apps and using the framework

Merge with OrchestraAI -> this is the main goal, allow the orchestration brain decide how toconsturct apps to cheive its goals, it can create its own tools, and use the tools it has to build new tools, and then use those tools to build apps, and then use those apps to build more apps, and so on and so forth, until it has built the tools it needs to build the apps it needs to build to achieve its goals.


DONE:

None issue in streaming printing. -- fixed by adding a check for None in the print function.
Check finetuning validation works. -- All good.
Logging is added for printouts so help debug app flow.
Fixed user message not being saved.
Implemented streaming with function calls.



