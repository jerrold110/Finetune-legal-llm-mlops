# Finetune-legal-llm-mlops

## Components
### Data versioning
Datasets are stored on S3 in a different path for each version so that files are not overwritten. MLRun versions datasets and allows for retrieval of the latest version as well as older versions. The version history of a dataset is viewable under the Datasets section, then click the circular arrow icon next to the Size column. 
Version labels are a string of the timestamp.

### Base model
An 8B model at 16bit weights (Llama3.1) requires the following amounts of memory for

8B model at native 16 bits:
8B * 2 bytes * 1.2
= 19.2 gb

8B model quantized to 8 bits:
8B * 1 bytes * 1.2
= 9.6 gb

14B model at 8 bits:
16.8

14-24B model at native 16 bits:
33.6 - 57.6

36B model at native 16 bits:
86.4

A 70B model at 16 bit weights:
70B * 2 * 1.2
= 168 gb

A 70B model at 16bit weights quantized to 8bit
70B * 1 * 1.2
= 84 gb

Further memory calculations:
https://docs.djl.ai/master/docs/serving/serving/docs/lmi/deployment_guide/instance-type-selection.html

## openai messages format is supported
https://docs.djl.ai/master/docs/serving/serving/docs/lmi/user_guides/chat_input_output_schema.html

### Model registry
The model files are stored in S3 while the registry is maintained in the local MLRun project

### Llama 3.1 model formats
https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/#prompt-template

https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B#prompt-format