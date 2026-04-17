from sagemaker.djl_inference.model import DJLModel
from sagemaker.utils import name_from_base
import os
# Loads AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY and MLRUN_AWS_ROLE_ARN
from dotenv import load_dotenv
load_dotenv()
import json
import boto3

# tested with async and batch config
# image = f"763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.36.0-lmi21.0.0-cu129" 
# tested with rolling batch config
image = f"763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.36.0-lmi24.0.0-cu129" 

role = os.getenv('AWS_ROLE_ARN')
HF_TOKEN = os.getenv("HF_TOKEN")

HF_MODEL_ID = "NousResearch/Hermes-4-14B-FP8"
MAX_MODEL_LEN = "8192" # max output tokens. total context window will be prompt + output tokens
INSTANCE_TYPE = "ml.g6e.2xlarge" # For Hermes-4-14B with tensor parallelism (model cannot fit on a single GPU)

# https://docs.djl.ai/master/docs/serving/serving/docs/lmi/user_guides/vllm_user_guide.html#advanced-vllm-configurations
lmi_batch_config = {
    "HF_MODEL_ID": HF_MODEL_ID,
    #"HF_REVISION": # commit or branch
    "HF_TOKEN": HF_TOKEN,
    "TENSOR_PARALLEL_DEGREE": "max",
    "OPTION_TENSOR_PARALLEL_DEGREE": "max",
    "MAX_BATCH_SIZE": "16", #default # 256, can overload memory "OPTION_MAX_ROLLING_BATCH_SIZE"
    "MAX_CONCURRENT_REQUESTS": "1000", #default queue size
    "OPTION_MAX_MODEL_LEN": MAX_MODEL_LEN,
    "OPTION_TRUST_REMOTE_CODE": "true",
    "OPTION_PREDICT_TIMEOUT":"300",
    # Rolling batch
    "SERVING_ENGINE": "Python",
    "OPTION_ROLLING_BATCH": "vllm",     # continuous batching
    "OPTION_MAX_ROLLING_BATCH_SIZE": "16", # instead of 64 for safety
    "OPTION_MAX_ROLLING_BATCH_PREFILL_TOKENS": "8000",
    # Enable lora
    "OPTION_ENABLE_LORA": "true",
    # Cache first part of the prompt, for repeated prompts
    "OPTION_ENABLE_PREFIX_CACHING": "true",
}

lmi_rt_config = {
    "HF_MODEL_ID": HF_MODEL_ID,
    "HF_TOKEN": HF_TOKEN,
    "OPTION_TRUST_REMOTE_CODE": "true",
    "SERVING_ENGINE": "Python",
    "OPTION_MAX_ROLLING_BATCH_SIZE":"16",
    "OPTION_MODEL_LOADING_TIMEOUT":"1800",
    "OPTION_MAX_MODEL_LEN": MAX_MODEL_LEN,
    "SERVING_FAIL_FAST":"true",
    "OPTION_ROLLING_BATCH":"disable",# disable vllm
    "OPTION_ASYNC_MODE":"true",
    "OPTION_ENTRYPOINT":"djl_python.lmi_vllm.vllm_async_service",
    "OPTION_PREDICT_TIMEOUT":"300", 
    #"OPTION_OUTPUT_FORMATTER": "json",  # This does not work
    "OPTION_TENSOR_PARALLEL_DEGREE": "max",
}

def deploy_endpoint_bat(source:str, name="lmi-batch-Hermes-14B"):
    """
    Deploy an endpoint with rolling batching and other inference optimisation strategies for high throughput. From a Hugging Face Hub url or S3 URI.

    """

    model = DJLModel(
    env=lmi_batch_config,
    role=role,
    image_uri=image,
    )

    predictor_batch = model.deploy(
        initial_instance_count=1,
        instance_type=INSTANCE_TYPE,
        endpoint_name=name_from_base(name),
        )

    endpoint_name = predictor_batch.endpoint_name
    print(f"Endpoint name is : {endpoint_name}")

    return predictor_batch


def invoke_endpoint_test(smr_client, endpoint_name:str, chatml_prompt:str):
    """
    
    Prompt should be formatted already. This has not been tested for responses that take longer than 1 minute. It is possible that this method may be necessary
    smr_client.invoke_endpoint_with_response_stream()

    This uses the client.invoke() method instead of endpoint.predict() because it enforces the encoding and decoding schema, HTTP headers.

    Prompt should be in the format:

    # chatml format
    chatml_prompt = (
        "<|im_start|>system\n"
        f"{system_prompt}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    chatml = {
        "inputs": chatml_prompt,
        "parameters": {
            "temperature": 0.2,
            "max_new_tokens": 3024, # Increase this up to your context limit if needed
            "do_sample": True       # Best practice to explicitly state this when adjusting temperature
        }
    }
    """

    response = smr_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(chatml_prompt)
    )

    response_dict = json.loads(response['Body'].read().decode('utf-8'))
    # remove the special token that is generated at the end
    response = response_dict['generated_text'].replace("<|im_end|>", "").strip()
    # parse a json formatted string into a dictionary
    parsed_response = json.loads(response, strict=False)

    return parsed_response

"""
# Invoke endpoint with streaming
resp = smr_client.invoke_endpoint_with_response_stream(
    EndpointName=endpoint_name,
    Body=json.dumps(body),
    ContentType="application/json",
)

# Process streaming response
for event in resp['Body']:
    if 'PayloadPart' in event:
        payload = event['PayloadPart']['Bytes'].decode()
        
        try:
            
            if payload.startswith('data: '):
                data = json.loads(payload[6:])  # Skip "data: " prefix
            else:
                data = json.loads(payload)
            
            token_count += 1
            if not first_token_received:
                ttft = time.time() - start_time
                first_token_received = True
            
            # Handle different streaming response formats
            if 'choices' in data and len(data['choices']) > 0:
                # Messages-compatible format
                if 'delta' in data['choices'][0] and 'content' in data['choices'][0]['delta']:
                    token_text = data['choices'][0]['delta']['content']
                    full_response += token_text
                    print(token_text, end='', flush=True)
            elif 'token' in data and 'text' in data['token']:
                # TGI format
                token_text = data['token']['text']
                full_response += token_text
                print(token_text, end='', flush=True)
        
        except json.JSONDecodeError:
            # Skip invalid JSON
            continue

"""

if __name__ == "__main___":
    smr_client = boto3.client('sagemaker-runtime')