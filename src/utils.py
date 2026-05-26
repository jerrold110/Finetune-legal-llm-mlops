"""
This file contains all the base modularised MLOps functions for use and reuse in different workflows.

Run evaluation on base model
Train new model and run evaluation
Register model
Deploy model endpoint
"""
# import sys
# print(sys.executable)
# exit(0)
# torch has to be imported first before transformers and sagemaker, becuase they import torch internally.
# this will initialise the DLLs first
import torch

import boto3
from botocore.config import Config
from sagemaker.djl_inference.model import DJLModel
from sagemaker.utils import name_from_base

from transformers import AutoTokenizer
from datasets import Dataset

import pyarrow as pa
import pyarrow.dataset as ds
import pandas as pd

import json
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
load_dotenv()
import os
import mlrun
##### Functions for evaluation and serving
############################
def deploy_djl_contbat(
        HF_REVISION="75875f970c359f89ad9e7d4dc86bf3c075c73c31"
        ):
    #-------------- Config -------------- 
    # DJL images. Should look at entrypoint files in this image.
    # https://aws.github.io/deep-learning-containers/reference/available_images/#djl-inference

    image = f"763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.36.0-lmi24.0.0-cu129" # this works with async and roll batch
    role = os.environ['MLRUN_AWS_ROLE_ARN']
    HF_MODEL_ID = "JerroldK/Hermes-4-14B-contract-extractor"
    INSTANCE_TYPE = "ml.g6e.2xlarge"
    BATCH_SIZE = "10"

    # https://docs.djl.ai/master/docs/serving/serving/docs/lmi/user_guides/vllm_user_guide.html#advanced-vllm-configurations
    # https://github.com/aws-samples/sagemaker-genai-hosting-examples/blob/main/Llama3.1/Benchmarking-LMI-containers-Llama3p1-Instruct.ipynb

    
    lmi_batch_config = {
        "HF_MODEL_ID": HF_MODEL_ID,
        #"HF_REVISION": # commit or branch
        "HF_TOKEN": os.environ['HF_TOKEN'], 
        "HF_REVISION": HF_REVISION,
        "SERVING_ENGINE": "Python", # https://docs.djl.ai/master/docs/serving/serving/docs/lmi/conceptual_guide/lmi_engine.html
        "OPTION_ROLLING_BATCH": "disable", #vllm
        "OPTION_ASYNC_MODE":"true",
        "TENSOR_PARALLEL_DEGREE": "1", # or "max"
        "OPTION_ENTRYPOINT":"djl_python.lmi_vllm.vllm_async_service", # this is from article
        "SERVING_FAIL_FAST":"true",
        "OPTION_QUANTIZE":"fp8",
        # This is new
        "VLLM_ATTENTION_BACKEND":"FLASH_ATTN", # TORCH_SDPA
        
        # The base model does not perform well when input is >8,000. Output is capped at 3,000
        "OPTION_MAX_MODEL_LEN":"11000", 
        
        # 64 is too aggressive for this instance. For 10k tokens, 32 is fine according to calculations, but still crashes
        "OPTION_MAX_ROLLING_BATCH_SIZE": BATCH_SIZE, 
        
        # Allow the endpoint to accept up to 200 requests into the queue at once
        "MAX_CONCURRENT_REQUESTS": BATCH_SIZE,
        "JOB_QUEUE_SIZE": '40',

        # The maximum time it will wait to receive a chunk of data from the Python backend. This is when waiting for previous batch to complete.
        "OPTION_PREDICT_TIMEOUT": "600",    # 10 mins
        "OPTION_MODEL_LOADING_TIMEOUT": "1200", # 20 mins
        
        # Important to enable this for caching the system prompt
        "OPTION_ENABLE_PREFIX_CACHING": "true",
        "OPTION_TRUST_REMOTE_CODE": "true",
        "OPTION_ENABLE_LORA": "false", # Enable for dynamic Lora adapters, reserves chunk of KV cache VRAM
    }
    print(lmi_batch_config)

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
                                  eval_data_key,
                                  eval_data_tag,
                                  prompt_key,
                                  prompt_tag,
                                  key):
    
    from src import utils_evaluate_model as evaluate_model
    # get input data handle data paths
    artifact = project.get_artifact(key=eval_data_key, tag=eval_data_tag)
    artifact_latest_s3_path = artifact.target_path

    print("s3_eval_path:")
    s3_eval_path = f"s3://legal-llama-data/evaluation/{key}/"
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
    
    tokenizer = AutoTokenizer.from_pretrained("JerroldK/Hermes-4-14B-contract-extractor")

    # get prompt template and prepare system prompt
    prompt_template = prompt_artifact.read_prompt()
    system_prompt = prompt_template[0]['content'] # unfortunately mlrun forces the openai messages format for prompt storage

    # get test data
    test_dataset = ds.dataset(
        source=artifact_latest_s3_path, 
        format="parquet")
    
    #small_test_dataset = test_dataset.to_table().to_pylist()[0:5]
    test_dataset = test_dataset.to_table().to_pylist()

    MINI_BATCH_SIZE = 10  # match API limit
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

##### Functions for training
############################

def prepare_train_datasets(project,
                           train_dataset,
                           train_dataset_tag,
                           val_dataset,
                           val_dataset_tag,
                           test_dataset,
                           test_dataset_tag,
                           prompt,
                           prompt_tag,
                           key):
    # train_uri = "store://datasets/finetune-legal-extractor/raw-proc-process-raw_train_data:latest"
    # validation_uri = "store://datasets/finetune-legal-extractor/raw-proc-process-raw_validation_data:latest"
    # key = datetime.now().strftime("%Y%m%d_%H%M")
    train_uri = project.get_artifact(key=train_dataset, tag=train_dataset_tag).target_path
    validation_uri = project.get_artifact(key=val_dataset, tag=val_dataset_tag).target_path
    test_uri = project.get_artifact(key=test_dataset, tag=test_dataset_tag).target_path

    ####################################################### HELPER FUNCTIONS
    def get_dataset(data_uri):
        data_pointer = mlrun.get_dataitem(data_uri)
        s3_path = data_pointer.url
        raw_dataset = ds.dataset(s3_path, format="parquet") # pyarrow FileSystemDataset

        return raw_dataset

    def get_sys_prompt(project,
                    prompt_key="contract_extractor_prompt",
                    prompt_tag="latest"):
        
        prompt_artifact = project.get_artifact(key=prompt_key, tag=prompt_tag)
        prompt_template = prompt_artifact.read_prompt()
        system_prompt = prompt_template[0]['content']

        return system_prompt

    def preprocess(batch, system_prompt, tokenizer, max_length=11000): # system + user <= 8000, assistant <= 3000
        
        system = system_prompt
        user = batch["text"]
        assistant = json.dumps(batch["inference"])

        # 1. Full conversation text (system + user + assistant response)
        # Chatml template
        full_messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
        full_text = tokenizer.apply_chat_template(
            full_messages, 
            tokenize=False, 
            add_generation_prompt=False,  # False: assistant response already in messages
            max_length=max_length
        )

        # 2. Prompt text: everything the model is allowed to *see*, not generate
        #    add_generation_prompt=True appends the assistant header (e.g. <|assistant|>)
        prompt_messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        # 3. Tokenize both
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

        # Sanity check: full_ids must start with prompt_ids
        if full_ids[:len(prompt_ids)] != prompt_ids:
            raise ValueError("Tokenization mismatch! Adjust your prompt split.")

        # 4. Build labels: mask prompt, keep response
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

        # 5. Truncate to max_length
        def pad_trim(ids):
            if len(ids) >= max_length:
                return ids[:max_length]
            return ids

        input_ids = pad_trim(full_ids)
        labels = pad_trim(labels)
        # Replace pad positions in labels with -100 so padding doesn't contribute loss
        labels = [lab if lab != tokenizer.pad_token_id else -100 for lab in labels]
        attention_mask = [1 if tok != tokenizer.pad_token_id else 0 for tok in input_ids]

        # format is lost during coversion from dict to datasetDict
        # return (torch.tensor(input_ids),
        #         torch.tensor(attention_mask),
        #         torch.tensor(labels))
        return (input_ids, attention_mask, labels, len(prompt_ids))

    def preprocess_and_format_to_tensor(raw_dataset, system_prompt, tokenizer):
        """
        Converts pyarrow datasets into datasets.arrow_dataset.Dataset
        """

        processed_data = {
            "input_ids":[],
            "attention_mask": [],
            "labels": []
        }
        count = 0
        for batch in raw_dataset.to_batches():
            # Process each pyarrow.RecordBatch
            print(f"Processing batch with {batch.num_rows} rows")
            for row in batch.to_pylist(): # 'row' is a standard Python dictionary
                input_id, attention_mask, label, token_length = preprocess(row,
                                                            system_prompt,
                                                            tokenizer) 
                # this is for testing on limited hardware because some samples go up to 12k tokens, causing OOM during training
                # most samples are less than 9000
                if token_length > 9000:
                    print(count, f"Token skipped. Length: {token_length}")
                    continue
                else:
                    processed_data['input_ids'].append(input_id)
                    processed_data['attention_mask'].append(attention_mask)
                    processed_data['labels'].append(label)

                    count += 1
                    print(count, f"Token length: {token_length}")

        processed_data = Dataset.from_dict(processed_data)
        #processed_data.set_format('torch', columns=['input_ids', 'attention_mask', 'labels']) # convert to pytorch tensors
        print(type(processed_data))
        return processed_data
    
    def simple_process(raw_dataset):
        """
        Converts pyarrow datasets into datasets.arrow_dataset.Dataset
        """
        processed_data = {
            "text":[],
            "inference": []
        }
        count = 0
        for batch in raw_dataset.to_batches():
            # Process each pyarrow.RecordBatch
            print(f"Processing batch with {batch.num_rows} rows")
            for row in batch.to_pylist():
                processed_data['text'].append(row['text'])
                processed_data['inference'].append(row['inference'])

                count += 1
                print(count)

        processed_data = Dataset.from_dict(processed_data)
        return processed_data
    #######################################################

    # Load datasets
    pointer_train = mlrun.get_dataitem(train_uri)
    train_pa = get_dataset(pointer_train.url)

    pointer_validation = mlrun.get_dataitem(validation_uri)
    validation_pa = get_dataset(pointer_validation.url)

    pointer_test = mlrun.get_dataitem(test_uri)
    test_pa = get_dataset(pointer_test.url)

    # Get system prompt
    system_prompt = get_sys_prompt(project,
                                   prompt,
                                   prompt_tag)

    # Get tokenizer
    tokenizer = AutoTokenizer.from_pretrained("JerroldK/Hermes-4-14B-contract-extractor")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare paths
    print("s3_eval_path:")
    s3_input_path = f"s3://legal-llama-data/training/{key}/"
    s3_train_input_path = s3_input_path + "train"
    s3_validation_input_path = s3_input_path + "validation"
    s3_test_input_path = s3_input_path + "test"
    print(s3_train_input_path, s3_validation_input_path, s3_test_input_path)

    # Upload raw train data
    test_dd = simple_process(test_pa)
    test_dd.save_to_disk(s3_test_input_path)
    print("Test data processed and uploaded")

    # Preprocess and upload data
    val_dd = preprocess_and_format_to_tensor(validation_pa, system_prompt, tokenizer)
    val_dd.save_to_disk(s3_validation_input_path)
    print("Validation data preprocessed and uploaded")

    train_dd = preprocess_and_format_to_tensor(train_pa, system_prompt, tokenizer)
    train_dd.save_to_disk(s3_train_input_path)
    print("Train data preprocessed and uploaded")

    return s3_validation_input_path, s3_train_input_path, s3_test_input_path

def train_model_get_outputs(
    key,
    model_repo,
    model_revision,
    epochs,
    batch_grad_accumulation,
    learning_rate,
    lora_r,
    lora_alpha,
    early_stopping_threshold,
):
    from sagemaker.huggingface import HuggingFace
    import matplotlib.pyplot as plt
    import io

    hftoken = os.environ['HF_TOKEN']
    iam = os.environ['MLRUN_AWS_ROLE_ARN']
    aws_no = os.environ['AWS_NO']
    # print(hftoken); exit(0)
    hyperparameters={
        'model_repo':model_repo,
        'model_revision':model_revision,
        'hftoken':hftoken,
        # This is only a small fraction of the parameters, but this is all I would change for my training strategy. This already produces very good training loss results
        'epochs':epochs,
        'batch_grad_accumulation':batch_grad_accumulation,
        'learning_rate':learning_rate,
        'lora_r':lora_r,
        'lora_alpha':lora_alpha,
        'early_stopping_threshold':early_stopping_threshold,
        'key':key
    }

    # Parallelism config. Currently only data parallelism
    distribution = {
        "torch_distributed": {
            "enabled": True
        }
    }

    bucket_name = 'legal-llama-data'
    
    huggingface_estimator = HuggingFace(entry_point='train_multi.py',
                                base_job_name='sm-hf-train',
                                source_dir='./scripts',     # should this be from /notebook or from /src? because the working dir should be /notebook
                                instance_type='ml.g6e.12xlarge',#
                                instance_count=1,
                                ###### max_wait should be equal to or greater than max_run in seconds
                                use_spot_instances=True,
                                max_wait=60*120,  # maximum time allowed for wait + run
                                max_run=60*90,   # maximum time allowed to run
                                checkpoint_s3_uri=f's3://legal-llama-data/training/{key}/checkpoints',
                                ######
                                role=iam,
                                py_version='py311', # why is this required if the image states the version already
                                image_uri=f'{aws_no}.dkr.ecr.us-east-1.amazonaws.com/smhf-torch2.5.1-flash-trans5.3.0-gpul4-py311-cu124:latest',
                                hyperparameters=hyperparameters,
                                distribution=distribution
                                )
    
    # Add a dummy isatty method so SageMaker doesn't crash
    import sys
    if not hasattr(sys.stdout, 'isatty'):
        sys.stdout.isatty = lambda: False
    print("⚠️Hugging Face estimator training job starting...")
    huggingface_estimator.fit()

    print("⚠️Getting loss data and commit id")
    # Get artifacts from training, and save loss curve as png on S3
    s3_client = boto3.client('s3')
    s3_lh_file_path = f'training/{key}/model_logs/training_history.json'
    s3_hfid_file_path = f'training/{key}/hfh_commit/commit_oid.txt'

    commit_oid = s3_client.get_object(
        Bucket='legal-llama-data',
        Key=s3_hfid_file_path,
    )

    commit_oid = commit_oid["Body"].read().decode("utf-8").strip()

    dictionary = s3_client.get_object(
        Bucket='legal-llama-data',
        Key=s3_lh_file_path ,
    )

    log_data = json.loads(
            dictionary["Body"].read().decode("utf-8")
        )

    train_loss = []
    train_steps = []
    eval_loss = []
    eval_steps = []

    for entry in log_data:
        if 'loss' in entry:
            train_loss.append(entry['loss'])
            train_steps.append(entry['step'])
            
        # Evaluation loss is typically logged under 'eval_loss'
        elif 'eval_loss' in entry:
            eval_loss.append(entry['eval_loss'])
            eval_steps.append(entry['step'])

    print("⚠️PLotting loss graph")
    plt.figure(figsize=(10, 6))
    plt.plot(train_steps, train_loss, marker='x', label='Training Loss', color='blue')

    # Only plot eval loss if it exists
    if eval_loss:
        plt.plot(eval_steps, eval_loss, marker='x', label='Validation Loss', color='orange')

    plt.title('Training and Validation Loss Curves')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    #plt.show()

    # 1. Save plot to an in-memory buffer
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight')

    # Reset the buffer's file pointer to the beginning so boto3 can read it
    img_buffer.seek(0)

    s3_graph_file_path = f'training/{key}/model_logs/training_curve.png'

    try:
        s3_client.upload_fileobj(img_buffer, bucket_name, s3_graph_file_path)
        print(f"Successfully saved plot to s3://{bucket_name}/{s3_graph_file_path}")
    except Exception as e:
        print(f"Failed to upload to S3: {e}")
    finally:
        # Clean up memory
        img_buffer.close()
        plt.close()

    return commit_oid, log_data