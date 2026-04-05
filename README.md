# Finetune-legal-llm-mlops

## Components
### Data versioning
Datasets are stored on S3. Versioning solution is still being researched

### Base model
Retrieve from S3 or HF directly

An 8B model at 16bit weights (Llama3.1) requires the following amounts of memory for

Inference at native 16 bits:
8B * 2 bytes * 1.2
= 19.2 gb

Inference quantized to 8 bits:
8B * 1 bytes * 1.2
= 9.6 gb

Further memory calculations:
https://docs.djl.ai/master/docs/serving/serving/docs/lmi/deployment_guide/instance-type-selection.html

### Model registry
The model files are stored in S3 while the registry is maintained in the local MLRun project