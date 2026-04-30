"""
This file contains all the base modularised MLOps functions for use and reuse in different workflows.

Run evaluation on existing model
Train new model and run evaluation
Register model
Deploy model
"""

from sagemaker.djl_inference.model import DJLModel
from sagemaker.utils import name_from_base
from concurrent.futures import ThreadPoolExecutor
import pyarrow.dataset as ds
from botocore.config import Config
from transformers import AutoTokenizer
import pyarrow as pa
import pandas as pd
import boto3
from datetime import datetime
import json

from src import utils_evaluate_model as evaluate_model


from dotenv import load_dotenv
load_dotenv()
import os


def deploy_djl_contbat(
        HF_REVISION="ed47684b01a083f4129d367e1a55f2c1371ba5e3"
        ):
    #-------------- Config -------------- 
    image = f"763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.36.0-lmi24.0.0-cu129" # this works with batch config
    role = "arn:aws:iam::975373241930:role/MLRunLLProj"
    HF_MODEL_ID = "JerroldK/Hermes-4-14B-FP8-legal-contract"
    INSTANCE_TYPE = "ml.g6e.2xlarge"    # 24 gb

    # https://docs.djl.ai/master/docs/serving/serving/docs/lmi/user_guides/vllm_user_guide.html#advanced-vllm-configurations
    lmi_batch_config = {
        "HF_MODEL_ID": HF_MODEL_ID,
        #"HF_REVISION": # commit or branch
        "HF_TOKEN": os.environ['HF_TOKEN'], 
        "HF_REVISION": HF_REVISION,
        "SERVING_ENGINE": "Python",
        "OPTION_ROLLING_BATCH": "disable", #vllm
        "OPTION_ASYNC_MODE":"true",
        "TENSOR_PARALLEL_DEGREE": "max", # or "max"
        "OPTION_ENTRYPOINT":"djl_python.lmi_vllm.vllm_async_service",
        "SERVING_FAIL_FAST":"true",
        
        # The base model does not perform well when input is >8,000. Output is capped at 3,000
        "OPTION_MAX_MODEL_LEN": "11000", 
        
        # 64 is too aggressive for this instance. For 10k tokens, 32 is fine according to calculations, but still crashes
        "OPTION_MAX_ROLLING_BATCH_SIZE": "15", 
        
        # Allow the endpoint to accept up to 200 requests into the queue at once
        "MAX_CONCURRENT_REQUESTS": "15",

        # The maximum time it will wait to receive a chunk of data from the Python backend. This is when waiting for previous batch to complete.
        "OPTION_PREDICT_TIMEOUT": "600",    # 10 mins
        "OPTION_MODEL_LOADING_TIMEOUT": "1200", # 20 mins
        
        # Important to enable this for caching the system prompt
        "OPTION_ENABLE_PREFIX_CACHING": "true",
        #"OPTION_QUANTIZE": "fp8", 
        "OPTION_TRUST_REMOTE_CODE": "true",
        "OPTION_ENABLE_LORA": "false", # Enable for dynamic Lora adapters, reserves chunk of KV cache VRAM
    }

    # About 10-12 mins if successful
    model = DJLModel(
        env=lmi_batch_config,
        role=role,
        image_uri=image,
        )

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=INSTANCE_TYPE,
        endpoint_name=name_from_base("lmi-batch-Hermes-14B-FP8"),
        )

    endpoint_name = predictor.endpoint_name
    print(endpoint_name)
    return predictor, endpoint_name

def soft_search_json(inference):
    start = inference.find('{')
    end = inference.rfind('}')

    return start, end

# def soft_search_json(inference):
#     start = inference.find('{')
#     end = inference.rfind('}')

#     return start, end

def invoke_model(smr_client, sys_prompt, parameters, contract_data, endpoint_name, tokenizer, id=""):
    """
    id is an optional identifier to track to inhouse data
    """
    # Format into the chatml template
    chatml_prompt = (
    "<|im_start|>system\n"
    f"{sys_prompt}<|im_end|>\n"
    "<|im_start|>user\n"
    f"{contract_data}<|im_end|>\n"
    "<|im_start|>assistant\n"
    )

    chatml = {
    "inputs": chatml_prompt,
    "parameters": parameters
    }

    # invoke endpoint with response stream
    #print(f"Beginning invocation stream on id {id}")
    try:
        resp = smr_client.invoke_endpoint_with_response_stream(
            EndpointName=endpoint_name,
            Body=json.dumps(chatml),
            ContentType="application/json",
            Accept="application/json",
        )

        full_response = ""
        tokens = tokenizer.encode(full_response)
        if len(tokens) > 3900:
            print(f"Warning: Document {id} response is over 3900 tokens")

        # Process streaming response
        for event in resp['Body']:
            if 'PayloadPart' in event:
                payload = event['PayloadPart']['Bytes'].decode('utf-8')
                full_response += payload
    except Exception as e:
        print(f"Error with doc {id} ", e)
    #print(f"Completed invocation stream on id {id}")
    try:
        full_response_dict = json.loads(full_response)
    except Exception as e:
        print(f"Error Document {id} during json.load() line 51:", e)
        #print(full_response)
        return -1

    # At this point we have not reached LLM generated tokens yet
    inference = full_response_dict['generated_text']
    # if "<|im_end|>" in inference:   # this should not happen if we enforce json formatter
    #     inference = inference.replace("<|im_end|>", "").strip()
    # else:
    #     inference = inference.strip()
    
    try:
        # start, end = soft_search_json(inference)
        # inference = inference[start:end+1]
        inference_dict = json.loads(inference)['hypotheses']
    except Exception as e:
        print(f"Error Document {id} during json.load() line 67:", e)
        print(inference)
        return -1

    return inference_dict

def process_single_row_testdata(row, sys_prompt, parameters, endpoint_name, tokenizer):
    # Create a new session and client
    custom_config = Config(
        read_timeout=900, # 15 mins
        connect_timeout=900,
        retries={"max_attempts": 1},
        max_pool_connections=1
    )
    thread_session = boto3.Session()
    smr_client = thread_session.client('sagemaker-runtime', config=custom_config)

    # Extract data from row
    document_id = row['document_id']
    contract_data = row['text']
    reference_dict = row['inference'] # list of dictionaries
    
    # ignore inference if contract_data is too long
    tokens = tokenizer.encode(contract_data)
    print(f"Document {document_id} is {len(tokens)} tokens\n")
    if len(tokens) > 8000:
        print(f"Document {document_id} is over 8000 tokens and ignored")
        return -1

    # make inference to model
    inference_dict = invoke_model(smr_client, sys_prompt, parameters, contract_data, endpoint_name, tokenizer, id=document_id)
    #print(inference_dict)
    if inference_dict == -1:
        return -1

    try:
        document_level_metrics = evaluate_model.document_level_metrics(reference_dict, inference_dict)
    except Exception as e:
        print("Error in calculating document_level_metrics:", e)
        print(inference_dict)
    #print(document_level_metrics)
    # return artifacts 
    return {'contract_data': contract_data,
            "inference_dict": inference_dict,
            "reference_dict": reference_dict,
            "document_level_metrics": document_level_metrics}

def process_multiple_row_testdata(project,
                                  endpoint_name,
                                  eval_data_key="raw-proc-process-raw_test_data",
                                  eval_data_tag="latest",
                                  prompt_key="contract_extractor_prompt",
                                  prompt_tag="latest"):
    # get input data handle data paths
    artifact = project.get_artifact(key=eval_data_key, tag=eval_data_tag)
    artifact_latest_s3_path = artifact.target_path

    key = datetime.now().strftime("%Y%m%d_%H%M")
    print("s3_eval_path:")
    s3_eval_path = f"s3://legal-llama-data/evaluation/{key}/"
    print()
    #s3_eval_output_path = s3_eval_path + "output.parquet"
    print(s3_eval_path)
    #print(s3_eval_output_path)

    # get prompt and invoc config
    prompt_artifact = project.get_artifact(key=prompt_key, tag=prompt_tag)
    prompt_tag = prompt_artifact.to_dict()['spec']['producer']['tag']
    invoc_config = prompt_artifact.to_dict()['spec']['invocation_config']
    print("invoc_config:")
    print(invoc_config)
    print()
    
    tokenizer = AutoTokenizer.from_pretrained("JerroldK/Hermes-4-14B-FP8-legal-contract")

    # get prompt template and prepare system prompt
    prompt_template = prompt_artifact.read_prompt()
    system_prompt = prompt_template[0]['content'] # unfortunately mlrun forces the openai messages format for prompt storage

    # get test data
    test_dataset = ds.dataset(
        source=artifact_latest_s3_path, 
        format="parquet")
    
    #small_test_dataset = test_dataset.to_table().to_pylist()[0:5]
    test_dataset = test_dataset.to_table().to_pylist()[:15]

    MINI_BATCH_SIZE = 15  # match API limit
    all_results = {"contract_data": [],
                   "inference_dict": [],
                   "reference_dict": [],
                   "document_level_metrics": []}

    # Run the inferences over each mini-batch in parallel
    for i in range(0, len(test_dataset), MINI_BATCH_SIZE):

        # Prepare the minibatch
        mini_batch = test_dataset[i : i + MINI_BATCH_SIZE]
        batch_number = (i // MINI_BATCH_SIZE) + 1
        print(f"--- Starting Mini-Batch {batch_number} ({len(mini_batch)} requests) ---")
    
        with ThreadPoolExecutor(max_workers=MINI_BATCH_SIZE) as executor:
            mbatch_results = list(
                executor.map(lambda x: process_single_row_testdata(x,
                                                                system_prompt,
                                                                invoc_config,
                                                                endpoint_name,
                                                                tokenizer
                                                                )
                            , mini_batch)
            )
            # remove invalid entries because the contract_data is too long
            mbatch_results = [x for x in mbatch_results if x != -1]
            #all_results.extend(mbatch_results)
            # loop over mbatch_results and append to all_results
            for i in mbatch_results:
                all_results["contract_data"].append(i["contract_data"])
                all_results["inference_dict"].append(i["inference_dict"])
                all_results["reference_dict"].append(i["reference_dict"])
                all_results["document_level_metrics"].append(i["document_level_metrics"])
            print(f"Mini-Batch {batch_number} completed successfully.")

    dataset_metrics = evaluate_model.dataset_level_metrics(all_results["document_level_metrics"])

    # Write all data to S3 output path
    # Ensure your AWS credentials are configured in your environment
    table = pa.table(all_results)
    ds.write_dataset(
        table, 
        base_dir=s3_eval_path,
        basename_template="inference_output{i}.parquet",
        format="parquet"
    )

    pd_dataset_metrics = pd.DataFrame(dataset_metrics, index=[range(len(dataset_metrics))])
    pd_dataset_metrics.to_json(f"{s3_eval_path}metrics.json", orient="records", lines=True)

    print("Inference results and metrics written to S3 at:")
    print(s3_eval_path)
    
    return dataset_metrics, s3_eval_path

