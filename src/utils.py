"""
This file contains all the modularised MLOps components for use and reuse in different MLOps workflows/pipelines.

Each MLOps pipeline is a combination of different components strung together to create a unique pipeline.

Internal Pipelines:
Experiment: Run evaluation on base model
Experiment: Train new model and run evaluation on fine-tuned model
Register model in model registry

External pipelines:
Deploy SM model endpoint
Update SM model endpoint
Deploy inference component adapter (LoRA adapter)
Update Inference component adapter (Routing gateway refer to docs)

Pipelines not in this file:
Register train/validation/test datasets
Data processing with update operations on datasets
Register prompt template and invocation configuration
Reinforcement learning fine-tuning pipelines

"""

# import sys
# print(sys.executable)
# exit(0)
# torch has to be imported first before transformers and sagemaker, becuase they import torch internally.
# this will initialise the DLLs first
import torch

import boto3
from botocore.config import Config

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

from src import utils_evaluate_model as evaluate_model

# import utils_evaluate_model as evaluate_model # for tests

# ========================================
# Functions for training and evaluation
# ========================================


def prepare_notrain_datasets(
    project,
    test_dataset,
    test_dataset_tag,
    prompt,
    prompt_tag,
    key,
):
    # key = datetime.now().strftime("%Y%m%d_%H%M")

    test_uri = project.get_artifact(key=test_dataset, tag=test_dataset_tag).target_path

    ####################################################### HELPER FUNCTIONS
    def get_dataset(data_uri):
        data_pointer = mlrun.get_dataitem(data_uri)
        s3_path = data_pointer.url
        raw_dataset = ds.dataset(s3_path, format="parquet")  # pyarrow FileSystemDataset

        return raw_dataset

    def get_sys_prompt(
        project, prompt_key="contract_extractor_prompt", prompt_tag="latest"
    ):

        prompt_artifact = project.get_artifact(key=prompt_key, tag=prompt_tag)
        prompt_template = prompt_artifact.read_prompt()
        system_prompt = prompt_template[0]["content"]

        return system_prompt

    def simple_process(raw_dataset, system_prompt, tokenizer, max_length=11000):
        """
        Converts pyarrow datasets into datasets.arrow_dataset.Dataset
        """
        processed_data = {"text": [], "inference": []}
        count = 0
        for batch in raw_dataset.to_batches():
            # Process each pyarrow.RecordBatch
            print(f"Processing batch with {batch.num_rows} rows")
            for row in batch.to_pylist():
                full_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": row["text"]},
                ]
                full_text = tokenizer.apply_chat_template(
                    full_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    max_length=max_length,
                )

                processed_data["text"].append(full_text)
                processed_data["inference"].append(row["inference"])

                count += 1
                print(count)

        processed_data = Dataset.from_dict(processed_data)
        return processed_data

    #######################################################

    # Load datasets
    pointer_test = mlrun.get_dataitem(test_uri)
    test_pa = get_dataset(pointer_test.url)

    # Get system prompt
    system_prompt = get_sys_prompt(project, prompt, prompt_tag)

    # Get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "JerroldK/Hermes-4-14B-contract-extractor"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare paths
    print("s3_eval_path:")
    s3_input_path = f"s3://legal-llama-data/notrain/{key}/"
    s3_test_input_path = s3_input_path + "test"
    print(s3_test_input_path)

    # Upload raw train data
    test_dd = simple_process(test_pa, system_prompt, tokenizer)
    test_dd.save_to_disk(s3_test_input_path)
    print("Test data processed and uploaded")

    return s3_test_input_path


def evaluate_model_base(
    key,
    project,
    model_repo,
    model_revision,
    prompt,
    prompt_tag,
):
    hftoken = os.environ["HF_TOKEN"]
    iam = os.environ["MLRUN_AWS_ROLE_ARN"]

    from sagemaker.pytorch import PyTorch

    # import sagemaker

    # If you define a session, it uses this instead of the default us-east-1
    # boto_session = boto3.Session(region_name="us-east-2")
    # sm_session = sagemaker.Session(boto_session=boto_session)
    # print("⚠️Us-east-2 sage session created")

    # Run evluation script as a training job, calculate metrics and upload to S3
    prompt_config = project.get_artifact(key=prompt, tag=prompt_tag).to_dict()["spec"][
        "invocation_config"
    ]

    _temp, _topp, _max_tok = (
        prompt_config["temperature"],
        prompt_config["top_p"],
        prompt_config["max_new_tokens"],
    )

    hyperparameters = {
        "model_repo": model_repo,
        "model_revision": model_revision,
        "hftoken": hftoken,
        "key": key,
        "prompt_temp": _temp,
        "prompt_topp": _topp,
        "prompt_max_tok": _max_tok,
    }
    print(hyperparameters)

    estimator = PyTorch(
        entry_point="quant_eval_vllm_base.py",
        base_job_name="sm-hf-basemodel-eval",
        source_dir="../src/scripts/evaluate",
        instance_type="ml.g6e.4xlarge",
        instance_count=1,
        ###### max_wait should be equal to or greater than max_run in seconds
        use_spot_instances=True,
        max_wait=60 * 60,  # maximum time allowed for wait + run
        max_run=60 * 45,  # maximum time allowed to run
        ######
        role=iam,
        py_version="py311",  # why is this required if the image states the version already
        image_uri="763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.10.0-gpu-py313-cu130-ubuntu22.04-sagemaker",
        hyperparameters=hyperparameters,
        # sagemaker_session=sm_session
    )
    import sys

    if not hasattr(sys.stdout, "isatty"):
        sys.stdout.isatty = lambda: False

    print("⚠️Pytorch estimator evaluation job starting...")
    estimator.fit()

    # Get the evaluation metrics from its path on S3, then return them both
    print("⚠️Getting loss data and commit id")
    s3_client = boto3.client("s3", region_name="us-east-1")
    bucket_name = "legal-llama-data"
    s3_met_file_path = f"notrain/{key}/metrics.json"

    dictionary = s3_client.get_object(Bucket=bucket_name, Key=s3_met_file_path)
    metric_data = json.loads(dictionary["Body"].read().decode("utf-8"))

    return metric_data, s3_met_file_path


def prepare_train_datasets(
    project,
    train_dataset,
    train_dataset_tag,
    val_dataset,
    val_dataset_tag,
    test_dataset,
    test_dataset_tag,
    prompt,
    prompt_tag,
    key,
):
    # train_uri = "store://datasets/finetune-legal-extractor/raw-proc-process-raw_train_data:latest"
    # validation_uri = "store://datasets/finetune-legal-extractor/raw-proc-process-raw_validation_data:latest"
    # key = datetime.now().strftime("%Y%m%d_%H%M")
    train_uri = project.get_artifact(
        key=train_dataset, tag=train_dataset_tag
    ).target_path
    validation_uri = project.get_artifact(
        key=val_dataset, tag=val_dataset_tag
    ).target_path
    test_uri = project.get_artifact(key=test_dataset, tag=test_dataset_tag).target_path

    ####################################################### HELPER FUNCTIONS
    def get_dataset(data_uri):
        data_pointer = mlrun.get_dataitem(data_uri)
        s3_path = data_pointer.url
        raw_dataset = ds.dataset(s3_path, format="parquet")  # pyarrow FileSystemDataset

        return raw_dataset

    def get_sys_prompt(project, prompt_key=prompt, prompt_tag=prompt_tag):

        prompt_artifact = project.get_artifact(key=prompt_key, tag=prompt_tag)
        prompt_template = prompt_artifact.read_prompt()
        system_prompt = prompt_template[0]["content"]

        return system_prompt

    def preprocess(
        batch,
        system_prompt,
        tokenizer,
        max_length=11000,
    ):  # system + user <= max_length. This should be changed to 8000

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
            max_length=max_length,
        )

        # 2. Prompt text: everything the model is allowed to *see*, not generate
        prompt_messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,  # this is important to append the assistant tag <|assistant|>
        )

        # 3. Tokenize both prompt_ids and full_ids
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

        # Sanity check: full_ids must start with prompt_ids
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError("Tokenization mismatch! Adjust your prompt split.")

        # 4. Build labels array: mask prompt tokens with -100, keep response
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]

        # 5. Truncate to max_length
        def pad_trim(ids):
            if len(ids) >= max_length:
                return ids[:max_length]
            return ids

        input_ids = pad_trim(full_ids)
        labels = pad_trim(labels)
        # Replace pad positions in labels with -100 so padding doesn't contribute loss
        labels = [lab if lab != tokenizer.pad_token_id else -100 for lab in labels]
        attention_mask = [
            1 if tok != tokenizer.pad_token_id else 0 for tok in input_ids
        ]

        # Format is lost during coversion from dict to datasetDict, do don't bother
        # return (torch.tensor(input_ids),
        #         torch.tensor(attention_mask),
        #         torch.tensor(labels))
        return (input_ids, attention_mask, labels, len(prompt_ids))

    def preprocess_and_format_to_tensor(
        raw_dataset,
        system_prompt,
        tokenizer,
    ):
        """
        Converts pyarrow datasets into datasets.arrow_dataset.Dataset
        """

        processed_data = {"input_ids": [], "attention_mask": [], "labels": []}
        count = 0
        for batch in raw_dataset.to_batches():
            # Process each pyarrow.RecordBatch
            print(f"Processing batch with {batch.num_rows} rows")
            for row in batch.to_pylist():  # 'row' is a standard Python dictionary
                input_id, attention_mask, label, token_length = preprocess(
                    row, system_prompt, tokenizer
                )
                # this is for testing on limited hardware because some samples go up to 12k tokens, causing OOM during training
                # most samples are less than 9000
                if token_length > 9000:
                    print(count, f"Token skipped. Length: {token_length}")
                    continue
                else:
                    processed_data["input_ids"].append(input_id)
                    processed_data["attention_mask"].append(attention_mask)
                    processed_data["labels"].append(label)

                    count += 1
                    print(count, f"Token length: {token_length}")

        processed_data = Dataset.from_dict(processed_data)
        # processed_data.set_format('torch', columns=['input_ids', 'attention_mask', 'labels']) # convert to pytorch tensors
        print(type(processed_data))
        return processed_data

    def simple_process(
        raw_dataset,
        system_prompt,
        tokenizer,
        max_length=11000,
    ):
        """
        Converts pyarrow datasets into datasets.arrow_dataset.Dataset
        """
        processed_data = {"text": [], "inference": []}
        count = 0
        for batch in raw_dataset.to_batches():
            # Process each pyarrow.RecordBatch
            print(f"Processing batch with {batch.num_rows} rows")
            for row in batch.to_pylist():
                full_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": row["text"]},
                ]
                full_text = tokenizer.apply_chat_template(
                    full_messages,
                    tokenize=False,
                    add_generation_prompt=True,  # this is important to append the assistant tag
                    max_length=max_length,
                )

                processed_data["text"].append(full_text)
                processed_data["inference"].append(row["inference"])

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
    system_prompt = get_sys_prompt(project, prompt, prompt_tag)

    # Get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "JerroldK/Hermes-4-14B-contract-extractor"
    )
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
    test_dd = simple_process(test_pa, system_prompt, tokenizer)
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

    hftoken = os.environ["HF_TOKEN"]
    iam = os.environ["MLRUN_AWS_ROLE_ARN"]
    aws_no = os.environ["AWS_NO"]
    # print(hftoken); exit(0)
    hyperparameters = {
        "model_repo": model_repo,
        "model_revision": model_revision,
        "hftoken": hftoken,
        # This is only a small fraction of the parameters, but this is all I would change for my training strategy. This already produces very good training loss results
        "epochs": epochs,
        "batch_grad_accumulation": batch_grad_accumulation,
        "learning_rate": learning_rate,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "early_stopping_threshold": early_stopping_threshold,
        "key": key,
    }
    """
    mpi_options = {
        "enabled" : True,
        "processes_per_host" : 4,   # 4 processes for 4 gpus in the instance
    }
    smp_options = {
        "enabled": True,
        "parameters": {
            "ddp": True, # Dist data parallelilsm
            # Sharded data parallelsim
            #"sharded_data_parallel_degree": 2, # This parameter specifies the number of GPUs over which the training state is sharded. Start small
            #"bf16": True
        }
    }

    distribution={
        "smdistributed": {"modelparallel": smp_options},
    }
    
    """

    # Parallelism config. Currently only data parallelism
    # Check help(HuggingFace) for documentation on the distribution parameter
    distribution = {"torch_distributed": {"enabled": True}}

    bucket_name = "legal-llama-data"

    huggingface_estimator = HuggingFace(
        entry_point="train_multi.py",
        base_job_name="sm-hf-train",
        source_dir="../src/scripts/training",  # working dir is /notebooks if called from a notebook
        instance_type="ml.g6e.12xlarge",
        instance_count=1,
        ###### max_wait should be equal to or greater than max_run in seconds
        use_spot_instances=True,
        max_wait=60 * 120,  # maximum time allowed for wait + run
        max_run=60 * 90,  # maximum time allowed to run
        checkpoint_s3_uri=f"s3://legal-llama-data/training/{key}/checkpoints",
        ######
        role=iam,
        py_version="py313",  # why is this required if the image states the version already
        image_uri=f"{aws_no}.dkr.ecr.us-east-1.amazonaws.com/smhf-torch2.5.1-flash-trans5.3.0-gpul4-py311-cu124:latest",
        hyperparameters=hyperparameters,
        distribution=distribution,
    )

    # Add a dummy isatty method so SageMaker doesn't crash
    import sys

    if not hasattr(sys.stdout, "isatty"):
        sys.stdout.isatty = lambda: False
    print("⚠️Hugging Face estimator training job starting...")
    huggingface_estimator.fit()

    print("⚠️Getting loss data and commit id")
    # Get artifacts from training, and save loss curve as png on S3
    s3_client = boto3.client("s3", region_name="us-east-1")
    s3_output_path = f"training/{key}"
    s3_lh_file_path = f"training/{key}/model_logs/training_history.json"
    s3_hfid_file_path = f"training/{key}/hfh_commit/commit_oid.txt"

    commit_oid = s3_client.get_object(
        Bucket="legal-llama-data",
        Key=s3_hfid_file_path,
    )

    commit_oid = commit_oid["Body"].read().decode("utf-8").strip()

    dictionary = s3_client.get_object(
        Bucket="legal-llama-data",
        Key=s3_lh_file_path,
    )

    log_data = json.loads(dictionary["Body"].read().decode("utf-8"))

    train_loss = []
    train_steps = []
    eval_loss = []
    eval_steps = []

    for entry in log_data:
        if "loss" in entry:
            train_loss.append(entry["loss"])
            train_steps.append(entry["step"])

        # Evaluation loss is typically logged under 'eval_loss'
        elif "eval_loss" in entry:
            eval_loss.append(entry["eval_loss"])
            eval_steps.append(entry["step"])

    # --------- Plot loss graph and upload to S3 ----------
    print("⚠️Plotting loss graph")
    plt.figure(figsize=(10, 6))
    plt.plot(train_steps, train_loss, marker="x", label="Training Loss", color="blue")

    # Only plot eval loss if it exists
    if eval_loss:
        plt.plot(
            eval_steps, eval_loss, marker="x", label="Validation Loss", color="orange"
        )

    plt.title("Training and Validation Loss Curves")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    # plt.show()

    # 1. Save plot to an in-memory buffer
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png", bbox_inches="tight")

    # Reset the buffer's file pointer to the beginning so boto3 can read it
    img_buffer.seek(0)

    s3_graph_file_path = f"training/{key}/model_logs/training_curve.png"
    # We should log the loss graph as well, but I skipped it this time

    try:
        s3_client.upload_fileobj(img_buffer, bucket_name, s3_graph_file_path)
        print(f"Successfully saved plot to s3://{bucket_name}/{s3_graph_file_path}")
    except Exception as e:
        print(f"Failed to upload to S3: {e}")
    finally:
        # Clean up memory
        img_buffer.close()
        plt.close()

    return commit_oid, log_data, s3_output_path


def evaluate_model_lora(
    key,
    project,
    model_repo,
    model_revision,
    prompt,
    prompt_tag,
    adapter_repo,
    adapter_revision,
):
    hftoken = os.environ["HF_TOKEN"]
    iam = os.environ["MLRUN_AWS_ROLE_ARN"]

    from sagemaker.pytorch import PyTorch

    # If you define a session, it uses this instead of the default us-east-1
    # boto_session = boto3.Session(region_name="us-east-2")
    # sm_session = sagemaker.Session(boto_session=boto_session)
    # print("⚠️Us-east-2 sage session created")

    # Run evluation script as a training job, calculate metrics and upload to S3
    prompt_config = project.get_artifact(key=prompt, tag=prompt_tag).to_dict()["spec"][
        "invocation_config"
    ]

    _temp, _topp, _max_tok = (
        prompt_config["temperature"],
        prompt_config["top_p"],
        prompt_config["max_new_tokens"],
    )

    hyperparameters = {
        "model_repo": model_repo,
        "model_revision": model_revision,
        "adapter_repo": adapter_repo,
        "adapter_revision": adapter_revision,
        "hftoken": hftoken,
        "key": key,
        "prompt_temp": _temp,
        "prompt_topp": _topp,
        "prompt_max_tok": _max_tok,
    }
    print(hyperparameters)

    estimator = PyTorch(
        entry_point="quant_eval_vllm_lora.py",
        base_job_name="sm-hf-lora-eval",
        source_dir="../src/scripts/evaluate",
        instance_type="ml.g6e.2xlarge",
        instance_count=1,
        ###### max_wait should be equal to or greater than max_run in seconds
        use_spot_instances=True,
        max_wait=60 * 60,  # maximum time allowed for wait + run
        max_run=60 * 45,  # maximum time allowed to run
        ######
        role=iam,
        py_version="py313",  # why is this required if the image states the version already
        image_uri="763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.10.0-gpu-py313-cu130-ubuntu22.04-sagemaker",
        hyperparameters=hyperparameters,
        # sagemaker_session=sm_session
    )
    import sys

    if not hasattr(sys.stdout, "isatty"):
        sys.stdout.isatty = lambda: False

    print("⚠️Pytorch estimator evaluation job starting...")
    estimator.fit()

    # Get the evaluation metrics from its path on S3, then return them both
    print("⚠️Getting loss data and commit id")
    s3_client = boto3.client("s3", region_name="us-east-1")
    bucket_name = "legal-llama-data"
    s3_met_file_path = f"training/{key}/evaluation/metrics.json"

    dictionary = s3_client.get_object(Bucket=bucket_name, Key=s3_met_file_path)
    metric_data = json.loads(dictionary["Body"].read().decode("utf-8"))
    s3_met_uri = f"s3://{bucket_name}/{s3_met_file_path}"

    return metric_data, s3_met_uri


# ========================================
# Functions for serving and testing in production
# ========================================
"""
https://docs.djl.ai/master/docs/serving/serving/docs/lmi/user_guides/vllm_user_guide.html#rolling-batch-configurations
https://docs.djl.ai/master/docs/serving/serving/docs/lmi/conceptual_guide/lmi_engine.html
https://github.com/aws-samples/sagemaker-genai-hosting-examples/blob/main/Llama3.1/Benchmarking-LMI-containers-Llama3p1-Instruct.ipynb
https://docs.djl.ai/master/docs/serving/plugins/secure-mode/index.html#security-controls
https://docs.djl.ai/master/docs/serving/serving/docs/lmi/conceptual_guide/lmi_engine.html


https://docs.djl.ai/master/docs/demos/aws/sagemaker/large-model-inference/sample-llm/multi_lora_gemma3_4b.html#invoke-base-ic
https://docs.djl.ai/master/docs/demos/aws/sagemaker/large-model-inference/sample-llm/stateful_inference_gemma3_4b_lora.html#invoke-adapter-ic

"""


def deploy_sm_lora_model(
    batch_size: str = "4",
    max_model_len: str = "7150",
    batch_tokens: str = "28600",
    instance: str = "ml.g6e.4xlarge",
):

    # from sagemaker.djl_inference.model import DJLModel
    from sagemaker import Model
    from sagemaker.compute_resource_requirements.resource_requirements import (
        ResourceRequirements,
    )
    from sagemaker.utils import name_from_base  # appends datetime
    from sagemaker.session import Session
    import sagemaker

    """
    Sagemaker model endpoints LMI is unstable and highly prone to memory leaks as performance tests have shown. Theoretically a batch size of up to 19 is possible, but tests show that a mediocre size of 4 is stable in production workloads - even with Flashinfer backend. Using g6e.2xlarge

    Based on the math at (6000+3000) and size of KV cache, 48GB VRAM, maximum batch size under memory limit can be 19; exluding activations/attention artifacts/cache updates. However throughput flattens as batch size increases and we reach a compute bottleneck during prefill (or some other hardware limitation).

    """

    """
    Model required an invocataion component adapter, unlike a base standalone model

    Autoscaling: 
    Scaling policies can be attach to a model after the endpoint has been deployed. https://docs.aws.amazon.com/sagemaker/latest/dg/endpoint-auto-scaling-add-code-apply.html

    Realtime endpoint adapters:
    https://docs.djl.ai/master/docs/demos/aws/sagemaker/large-model-inference/sample-llm/multi_lora_gemma3_4b.html#clean-up-resources
    https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints-adapt.html
    
    """

    region = "us-east-1"
    boto_session = boto3.Session(region_name=region)
    sagemaker_session = Session(boto_session=boto_session)

    # DJL images. Should look at entrypoint files in this image.
    # https://aws.github.io/deep-learning-containers/reference/available_images/#djl-inference
    image = f"763104351884.dkr.ecr.{region}.amazonaws.com/djl-inference:0.36.0-lmi24.0.0-cu129"  # this works with async and roll batch
    role = os.environ["MLRUN_AWS_ROLE_ARN"]
    HF_MODEL_ID = "JerroldK/Hermes-4-14B-contract-extractor"
    HF_REVISION = "75875f970c359f89ad9e7d4dc86bf3c075c73c31"
    INSTANCE_TYPE = instance
    print(INSTANCE_TYPE)

    BATCH_SIZE = batch_size
    MAX_MODEL_LEN = max_model_len
    BATCH_TOKENS = batch_tokens

    lmi_batch_config = {
        "HF_MODEL_ID": HF_MODEL_ID,
        "HF_TOKEN": os.environ["HF_TOKEN"],
        "HF_REVISION": HF_REVISION,
        "SERVING_ENGINE": "Python",  # https://docs.djl.ai/master/docs/serving/serving/docs/lmi/conceptual_guide/lmi_engine.html
        "OPTION_ROLLING_BATCH": "disable",  # vllm disable
        "OPTION_ASYNC_MODE": "true",
        # "max" if enable tensor parallelism
        # 1 enables data parallelism since 1 model per gpu
        "TENSOR_PARALLEL_DEGREE": "1",
        "OPTION_ENTRYPOINT": "djl_python.lmi_vllm.vllm_async_service",  # this is from article
        "OPTION_QUANTIZE": "fp8",
        "OPTION_KV_CACHE_DTYPE": "fp8",
        "OPTION_GPU_MEMORY_UTILIZATION": "0.95",
        # "OPTION_ENABLE_CHUNKED_PREFILL":"false",
        # "OPTION_ENABLE_PREFIX_CACHING":"false",
        # "OPTION_ENFORCE_EAGER":"true",
        "OPTION_MAX_MODEL_LEN": MAX_MODEL_LEN,  # Max input + output = 6000 + 3000, this is a little buggy because requests close to but under 9000 tokens exceed this hard limit
        "MAX_BATCH_SIZE": BATCH_SIZE,  # https://docs.djl.ai/master/docs/serving/serving/docs/lmi/user_guides/starting-guide.html
        "MAX_CONCURRENT_REQUESTS": "200",
        "OPTION_MAX_ROLLING_BATCH_SIZE": BATCH_SIZE,  # this is in the amazon articles for async serving
        # --- ADVANCED SETTINGS -----
        # https://docs.djl.ai/master/docs/serving/serving/docs/lmi/user_guides/vllm_user_guide.html#advanced-vllm-configurations
        "OPTION_MAX_NUM_BATCHED_TOKENS": BATCH_TOKENS,  # Limits the number of tokens that can be processed in a single step during prefill
        # https://docs.vllm.ai/en/v0.8.3/serving/env_vars.html
        "VLLM_ATTENTION_BACKEND": "FLASHINFER",  # TORCH_SDPA  FLASH_ATTN
        # The maximum time it will wait to receive a chunk of data from the Python backend. This is when waiting for previous batch to complete.
        "OPTION_PREDICT_TIMEOUT": str(60 * 15),
        "OPTION_MODEL_LOADING_TIMEOUT": str(60 * 20),
        "OPTION_TRUST_REMOTE_CODE": "true",
        "SERVING_FAIL_FAST": "true",
        "OPTION_ENABLE_LORA": "true",  # Enable for dynamic Lora adapters, reserves chunk of KV cache VRAM
        "OPTION_MAX_LORA_RANK": "16",
        "OPTION_PARALLEL_LOADING": "true",  # parallel model loading when loading multiple model workers, inc temp memory footprint
        "SERVING_JOB_QUEUE_SIZE": "500",  # Default is 1000
    }
    # THIS IS OUTDATED AND REQUIRES A DIFFERENT IMAGE. DO NOT USE
    # lmi_old_batch_config = {
    #     "HF_MODEL_ID": HF_MODEL_ID,
    #     "HF_TOKEN": os.environ['HF_TOKEN'],
    #     "HF_REVISION": HF_REVISION,

    #     "OPTION_ROLLING_BATCH": "vllm", #vllm disable
    #     "OPTION_ENFORCE_EAGER":"true",
    #     "TENSOR_PARALLEL_DEGREE": "1", # or "max" if enable tensor parallelism
    #     "OPTION_QUANTIZE":"fp8",
    #     "OPTION_KV_CACHE_DTYPE":"fp8",

    #     "OPTION_MAX_MODEL_LEN":MAX_MODEL_LEN, # Max input + output = 6000 + 3000, this is a little buggy because requests close to but under 9000 tokens exceed this hard limit
    #     "MAX_CONCURRENT_REQUESTS": "200",
    #     "OPTION_MAX_ROLLING_BATCH_SIZE":BATCH_SIZE, # this is in the amazon articles for async serving
    #     # --- ADVANCED SETTINGS -----
    #     # https://docs.djl.ai/master/docs/serving/serving/docs/lmi/user_guides/vllm_user_guide.html#advanced-vllm-configurations
    #     "OPTION_MAX_NUM_BATCHED_TOKENS": BATCH_TOKENS,# Limits the number of tokens that can be processed in a single step during prefill

    #     # https://docs.vllm.ai/en/v0.8.3/serving/env_vars.html
    #     "VLLM_ATTENTION_BACKEND":"FLASH_ATTN", # TORCH_SDPA  FLASH_ATTN FLASHINFER
    #     "OPTION_ENABLE_CHUNKED_PREFILL":"false",
    #     "OPTION_ENABLE_PREFIX_CACHING":"false",
    #     "OPTION_ASYNC_MODE":"false",

    #     # The maximum time it will wait to receive a chunk of data from the Python backend. This is when waiting for previous batch to complete.
    #     "OPTION_PREDICT_TIMEOUT": str(60*10),    # 10 mins
    #     "OPTION_MODEL_LOADING_TIMEOUT": str(60*20), # 20 mins
    #     "SERVING_FAIL_FAST":"true",

    #     "OPTION_ENABLE_LORA": "true", # Enable for dynamic Lora adapters, reserves chunk of KV cache VRAM
    #     "OPTION_MAX_LORA_RANK": "16",
    #     "OPTION_PARALLEL_LOADING": "true", # parallel model loading when loading multiple model workers, inc temp memory footprint
    #     "SERVING_JOB_QUEUE_SIZE": '500', # Default is 1000
    # }

    for k, v in lmi_batch_config.items():
        print(k, v)

    # About 10-12 mins if successful
    model = Model(
        env=lmi_batch_config,
        role=role,
        image_uri=image,
        sagemaker_session=sagemaker_session,
    )

    endpoint_name = name_from_base("lmi-Hermes-FP8")
    base_ic_name = "base-" + endpoint_name

    print("Deploying model endpoint...")
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=INSTANCE_TYPE,
        endpoint_name=endpoint_name,
        endpoint_type=sagemaker.enums.EndpointType.INFERENCE_COMPONENT_BASED,
        inference_component_name=base_ic_name,
        resources=ResourceRequirements(
            requests={"num_accelerators": 1, "memory": 256, "copies": 1}
        ),  # Resource requirements for inference component object
    )

    # endpoint_name = predictor.endpoint_name # unavailable for lora endpoint
    print(f"✅ Deployed model (LoRA): {endpoint_name}, {base_ic_name}")

    # Attach autoscaling policy to base_ic_name, logs are under base_ic_name not endpoint_name

    return endpoint_name, base_ic_name


def deploy_sm_lora_adapter(key, endpoint_name, base_ic_name, adapter_revision):
    from huggingface_hub import HfFileSystem
    import tarfile
    import io
    import sagemaker
    from sagemaker.session import Session
    from sagemaker.utils import name_from_base  # appends datetime

    """
    
    """

    REGION = "us-east-1"
    BUCKET = "legal-llama-data"

    ADAPTER_ID = "JerroldK/H4-14b-contract-extractor-adapter"
    ADAPTER_FILENAME = "adapter.tar.gz"
    S3_KEY = f"Adapters/{key}/{ADAPTER_FILENAME}"
    S3_URI = f"s3://{BUCKET}/{S3_KEY}"

    ic_adapter_name = f'adapter-{name_from_base("lmi-Hermes-FP8")}'

    boto_session = boto3.Session(region_name="us-east-1")
    s3_client = boto_session.client("s3", region_name="us-east-1")
    sm_client = boto_session.client("sagemaker", region_name="us-east-1")
    sess = Session(boto_session=boto_session)

    # ------------ Compress and upload adapter to S3
    fs = HfFileSystem(token=os.environ["HF_TOKEN"])

    # Create an in-memory buffer for the tar archive
    tar_buffer = io.BytesIO()
    print(
        f"Fetching files from '{ADAPTER_ID}' {adapter_revision} and building {ADAPTER_FILENAME} in memory..."
    )

    # Open the tar buffer for writing in gzip mode
    with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:

        # fs.find() recursively gets all files in the Hugging Face repo
        for file_path in fs.find(path=ADAPTER_ID, revision=adapter_revision):

            # Read the file's bytes directly from Hugging Face
            with fs.open(file_path, "rb") as f:
                file_data = f.read()

            file_obj = io.BytesIO(file_data)

            # Strip the repo name from the path so it extracts cleanly without user/repo_name@hash/
            # relative_path = file_path.replace(f"{ADAPTER_ID}/", "")
            relative_path = "/".join(
                file_path.split("/")[2:]
            )  # "/".join() is for nested folders
            print(f"  -> Packaging: {file_path} as {relative_path}")
            # Create tar file metadata
            tarinfo = tarfile.TarInfo(name=relative_path)
            tarinfo.size = len(file_data)
            # Add the file to the in-memory tar archive
            tar.addfile(tarinfo, fileobj=file_obj)

    # Rewind the buffer to the beginning before uploading
    tar_buffer.seek(0)

    print(f"⚠️ Uploading directly to {S3_URI}...")

    # Stream the buffer directly to S3
    s3_client.upload_fileobj(tar_buffer, BUCKET, S3_KEY)

    print("✅ Success! Transfer complete without touching the disk.")

    # ------------- Create inference component for model endpoint
    sm_client = boto3.client(
        service_name="sagemaker", region_name="us-east-1"
    )  # not sagemaker-runtime
    sess = sagemaker.session.Session()
    # iam = os.environ['MLRUN_AWS_ROLE_ARN']

    adapter_inference = sm_client.create_inference_component(
        InferenceComponentName=ic_adapter_name,  # extension of endpoint_name
        EndpointName=endpoint_name,
        Specification={
            "BaseInferenceComponentName": base_ic_name,
            "Container": {"ArtifactUrl": S3_URI},
        },
    )
    sess.wait_for_inference_component(ic_adapter_name)

    print(
        f"✅ Created Adapter inference component {ic_adapter_name} for endpoint {endpoint_name} ARN: {adapter_inference['InferenceComponentArn']}"
    )

    return adapter_inference, ic_adapter_name


# ========================================
# Register model
# ========================================

# def register_model(
#     project,
#     experiment_run_uid,
#     version
#     ):

#     from mlrun.model import RunObject

#     # model-purpose-artifacts
#     model_key = "Hermes-4-14B-ContractExtractor-model-adapter"

#     # Initialize the MLRun DB client
#     db = mlrun.get_run_db()
#     run_dict = db.read_run(uid=experiment_run_uid, project="finetune-legal-extractor")

#     # Convert the dictionary to a RunObject for easier API access
#     run = RunObject.from_dict(run_dict)
#     run_parameters = run_dict['spec']['parameters']
#     run_metrics = run_dict['status']['results']
#     output = run.outputs['return'] # this is what was returned

#     # Pass in model_id, commit, hyperparameters, performance metrics
#     #version = datetime.now().strftime("%Y%m%d_%H%M")

#     model = project.log_model(
#                     key=f'{model_key}-{version}',
#                     tag="NA",
#                     metrics=run_metrics,
#                     parameters=run_parameters,
#                     framework="Hugging Face model with adapter",
#                     model_url="https://huggingface.co/JerroldK/H4-14b-contract-extractor-adapter",
#                     labels={"model": "Hermes-4-14B"},
#                     upload=False
#                     )
#     print('========== MODEL METADATA ==========')
#     print(model.tag)
#     print(model.labels)
#     print(model.model_url)
#     print(model.metrics)
#     print(model.parameters)

#     print(f"{model_key} model logged with version:\n {version}")

# ========================================
# Load test of model endpoint
# ========================================


def invoke_model(
    smr_client,
    sys_prompt,
    parameters,
    contract_data,
    endpoint_name,
    adapter_name,
    tokenizer,
    id="",
    max_prompt_l=5000,
):
    """
    change max_token_length parameter outside this function before passing parameters
    id is an optional identifier to track to inhouse data
    """
    # print(parameters)

    prompt = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": contract_data},
    ]

    chatml_prompt = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True,  # this is important to append the assistant tag
    )

    # ignore inference if contract_data is too long
    tokens = tokenizer.encode(chatml_prompt)
    print(f"Document {id} prompt length is {len(tokens)} tokens")
    if len(tokens) >= max_prompt_l:
        print(f"Document {id} prompt length is over {max_prompt_l} tokens and ignored")
        return -1

    chatml = {"inputs": chatml_prompt, "parameters": parameters}

    # invoke endpoint with response stream
    # print(f"Beginning invocation stream on id {id}")
    try:
        resp = smr_client.invoke_endpoint_with_response_stream(
            EndpointName=endpoint_name,
            Body=json.dumps(chatml),
            ContentType="application/json",
            Accept="application/json",
            InferenceComponentName=adapter_name,  # remove this line for base model endpoint invocation
        )
        # Process streaming response
        full_response = ""
        for event in resp["Body"]:
            if "PayloadPart" in event:
                payload = event["PayloadPart"]["Bytes"].decode("utf-8")
                full_response += payload

        tokens = tokenizer.encode(full_response)
        l_t = len(tokens)
        print(f"Doc {id} response length is {l_t} tokens ✅")
        # if l_t > 4900:
        #     print(f"Warning: Document {id} response is over 4900 tokens")
    except Exception as e:
        print(f"Error with doc {id} part A", e)

    try:
        full_response_dict = json.loads(full_response)
    except Exception as e:
        print(f"Error Document {id} during json.load() Part B:", e)
        print(full_response)
        return -1

    # At this point we have not reached LLM generated tokens yet
    # print("---->", full_response_dict)
    inference = full_response_dict["generated_text"]
    # if "<|im_end|>" in inference:   # this should not happen if we enforce json formatter
    #     inference = inference.replace("<|im_end|>", "").strip()
    # else:
    #     inference = inference.strip()

    try:
        # start, end = soft_search_json(inference)
        # inference = inference[start:end+1]
        inference_dict = json.loads(inference)["Hypotheses"]
    except Exception as e:
        print(f"Error Document {id} during json.load() part C:", e)
        print(inference)
        return -1

    return inference_dict


def process_single_row_testdata(
    row, sys_prompt, parameters, endpoint_name, adapter_name, tokenizer
):
    # Create a new session and client
    custom_config = Config(
        read_timeout=900,  # 15 mins
        connect_timeout=900,
        retries={"max_attempts": 1},
        max_pool_connections=1,
    )
    thread_session = boto3.Session()
    smr_client = thread_session.client(
        "sagemaker-runtime", config=custom_config, region_name="us-east-1"
    )

    # Extract data from row
    document_id = row["document_id"]
    contract_data = row["text"]
    reference_dict = row["inference"]  # list of dictionaries

    # make inference to model
    inference_dict = invoke_model(
        smr_client,
        sys_prompt,
        parameters,
        contract_data,
        endpoint_name,
        adapter_name,
        tokenizer,
        id=document_id,
    )
    # print(inference_dict)
    if inference_dict == -1:
        return -1

    try:
        document_level_metrics = evaluate_model.document_level_metrics(
            reference_dict, inference_dict
        )
    except Exception as e:
        print("Error in calculating document_level_metrics:", e)
        print(inference_dict)
    # print(document_level_metrics)
    # return artifacts
    return {
        "contract_data": contract_data,
        "inference_dict": inference_dict,
        "reference_dict": reference_dict,
        "document_level_metrics": document_level_metrics,
    }


def process_multiple_row_testdata(
    project,
    endpoint_name,
    adapter_name,
    eval_data_key,
    eval_data_tag,
    prompt_key,
    prompt_tag,
    key,
    max_output_l=2000,  # don't increase
    batchsize=5,
):

    # get input data handle data paths
    artifact = project.get_artifact(key=eval_data_key, tag=eval_data_tag)
    artifact_latest_s3_path = artifact.target_path

    # print("s3_eval_path:")
    # s3_eval_path = f"s3://legal-llama-data/evaluation/{key}/"
    # print(s3_eval_path)

    # get prompt and invoc config
    prompt_artifact = project.get_artifact(key=prompt_key, tag=prompt_tag)
    prompt_tag = prompt_artifact.to_dict()["spec"]["producer"]["tag"]
    invoc_config = prompt_artifact.to_dict()["spec"]["invocation_config"]
    print("invoc_config:")
    if invoc_config["max_new_tokens"] > max_output_l:
        invoc_config["max_new_tokens"] = max_output_l
        print(f"max_new_tokens is too high and reduced to {max_output_l}")

    # print(invoc_config)
    # print()

    tokenizer = AutoTokenizer.from_pretrained(
        "JerroldK/Hermes-4-14B-contract-extractor"
    )

    # get prompt template and prepare system prompt
    prompt_template = prompt_artifact.read_prompt()
    system_prompt = prompt_template[0][
        "content"
    ]  # unfortunately mlrun forces the openai messages format for prompt storage

    # get test data
    test_dataset = ds.dataset(source=artifact_latest_s3_path, format="parquet")
    # test the first 10 samples
    # test_dataset = test_dataset.to_table().to_pylist()[:10]

    # test 5 random samples
    import random

    testSize = 5
    test_dataset = test_dataset.to_table().to_pylist()
    start = random.randint(0, len(test_dataset) - testSize)
    test_dataset = test_dataset[start : start + testSize]

    MINI_BATCH_SIZE = batchsize  # or whatever, it goes to the queue
    all_results = {
        "contract_data": [],
        "inference_dict": [],
        "reference_dict": [],
        "document_level_metrics": [],
    }

    # Run the inferences over each mini-batch in parallel
    for i in range(0, len(test_dataset), MINI_BATCH_SIZE):

        # Prepare the minibatch
        mini_batch = test_dataset[i : i + MINI_BATCH_SIZE]
        batch_number = (i // MINI_BATCH_SIZE) + 1
        print(
            f"\n--- Starting Mini-Batch {batch_number} ({len(mini_batch)} requests) ---"
        )

        with ThreadPoolExecutor(max_workers=MINI_BATCH_SIZE) as executor:
            mbatch_results = list(
                executor.map(
                    lambda x: process_single_row_testdata(
                        x,
                        system_prompt,
                        invoc_config,
                        endpoint_name,
                        adapter_name,
                        tokenizer,
                    ),
                    mini_batch,
                )
            )
            # remove invalid entries because the contract_data is too long
            mbatch_results = [x for x in mbatch_results if x != -1]
            # all_results.extend(mbatch_results)
            # loop over mbatch_results and append to all_results
            for i in mbatch_results:
                all_results["contract_data"].append(i["contract_data"])
                all_results["inference_dict"].append(i["inference_dict"])
                all_results["reference_dict"].append(i["reference_dict"])
                all_results["document_level_metrics"].append(
                    i["document_level_metrics"]
                )
            print(f"Mini-Batch {batch_number} completed successfully.")

    dataset_metrics = evaluate_model.dataset_level_metrics(
        all_results["document_level_metrics"]
    )
    print(dataset_metrics)

    # Write all data to S3 output path
    # Ensure your AWS credentials are configured in your environment
    # table = pa.table(all_results)
    # ds.write_dataset(
    #     table,
    #     base_dir=s3_eval_path,
    #     basename_template="inference_output{i}.parquet",
    #     format="parquet"
    # )

    # pd_dataset_metrics = pd.DataFrame(dataset_metrics, index=[range(len(dataset_metrics))])
    # pd_dataset_metrics.to_json(f"{s3_eval_path}metrics.json", orient="records", lines=True)

    # print("Inference results and metrics written to S3 at:")
    # print(s3_eval_path)
    s3_eval_path = ""

    return dataset_metrics, s3_eval_path


# ========================================
# Update traffic gateway with rollback
# ========================================


def update_gateway_destination_sm(
    project,
    model_endpoint: str,
    model_adapter: str,
    template_uri: str,
    rolling: bool,
    deployment_color: str,
):

    # Get infra var names from terraform out
    import subprocess

    parent_dir = os.path.abspath("../Terraform/")  # this is called from notebook

    result = subprocess.run(
        ["terraform", "output", "-json"],
        capture_output=True,
        text=True,
        check=True,
        cwd=parent_dir,
    )
    outputs = json.loads(result.stdout)
    print(outputs)
    values = {key: data["value"] for key, data in outputs.items()}
    try:
        appconfig_app_id = values["appconfig_app_id"]
        appconfig_env_id = values["appconfig_env_id"]
        appconfig_confprof_cpid = values["appconfig_confprof_cpid"]
        appconfig_deploystrat_direct_id = values["appconfig_deploystrat_direct_id"]
        appconfig_deploystrat_rolling_id = values["appconfig_deploystrat_rolling_id"]
        iam_arn_appconfig_cw_malarm_rollback = values[
            "iam_appconfig_cw_malarm_rollback"
        ]

        # Arns of clouwdwatch metric aalarms
        cwa_1b = values["appconfig_cw_malarm_1b_arn"]
        cwa_1w = values["appconfig_cw_malarm_1w_arn"]

    except KeyError as e:
        raise KeyError(f"Missing required configuration key: {e.args[0]}") from None

    ac_client = boto3.client("appconfig", region_name="us-east-1")

    # Update the cloudwatch metric alarms attached to the AppConfig environment, so that rollback logic changes because of the new deployment color

    if deployment_color == "Black":
        monitors = [
            {"AlarmArn": cwa_1b, "AlarmRoleArn": iam_arn_appconfig_cw_malarm_rollback}
        ]
    elif deployment_color == "White":
        monitors = [
            {"AlarmArn": cwa_1w, "AlarmRoleArn": iam_arn_appconfig_cw_malarm_rollback}
        ]
    print("======> Updating AppConfig environment in prepration for deployment")
    print(appconfig_app_id, appconfig_env_id)
    response = ac_client.update_environment(
        ApplicationId=appconfig_app_id,
        EnvironmentId=appconfig_env_id,
        Monitors=monitors,
    )

    ################################
    # Update the free-form configuration profile read in lambda

    configuration = {
        "model_endpoint": model_endpoint,
        "model_adapter": model_adapter,
        "template_uri": template_uri,
        "deployment_color": deployment_color,
    }

    host_config_response = ac_client.create_hosted_configuration_version(
        ApplicationId=appconfig_app_id,
        ConfigurationProfileId=appconfig_confprof_cpid,
        Content=json.dumps(configuration).encode("utf-8"),
        ContentType="application/json",
        Description="AppConfig configuration profile for a rolling or direct deployment to be read in Lambda",
    )
    version_number = host_config_response["VersionNumber"]

    if rolling == False:
        deployment_strategy_id = appconfig_deploystrat_direct_id
    elif rolling == True:
        deployment_strategy_id = appconfig_deploystrat_rolling_id

    deploy = ac_client.start_deployment(
        ApplicationId=appconfig_app_id,
        EnvironmentId=appconfig_env_id,
        # deployment strategy
        DeploymentStrategyId=deployment_strategy_id,
        # configuration profile created on AppConfig hosted configuration profile
        ConfigurationProfileId=appconfig_confprof_cpid,
        ConfigurationVersion=str(version_number),
    )
    deployment_number = deploy["DeploymentNumber"]
    print("Deployment number:", deployment_number)
    for k, v in deploy.items():
        print(k, v)

    print("")
    if rolling:
        # Begin the 20 minute test
        print(
            "===> Deployment workflow finished, rolling update in progress, now monitoring new model in preparation for rollback"
        )
        print(
            "===> Now run the rolling_update_test_20mins() function manually to simulate load"
        )
        import time

        # Poll GetDeployment until it reaches a terminal state
        while True:
            deployment = ac_client.get_deployment(
                ApplicationId=appconfig_app_id,
                EnvironmentId=appconfig_env_id,
                DeploymentNumber=deployment_number,
            )

            state = deployment["State"]
            print(f"Deployment state: {state}")
            state = deployment["State"]
            event_log = deployment.get("EventLog", [])

            if state == "COMPLETE":
                print("✅Rolling update complete with no rollback")
                break

            elif state in ("ROLLING_BACK", "ROLLED_BACK", "REVERTED", "FAILED"):
                raise RuntimeError(
                    f"Deployment ended in state {state}. "
                    f"Latest event: {event_log[-1] if event_log else 'None'}"
                )

            time.sleep(10)
    else:
        print(
            "✅ Direct deployment finished, now monitoring new model in preparation for long-term model drift rollback (if it exists)"
        )


def test_lambda_few_rows(
    project,
    eval_data_key,
    eval_data_tag,
    l_client,
):

    # get input data handle data paths
    artifact = project.get_artifact(key=eval_data_key, tag=eval_data_tag)
    artifact_latest_s3_path = artifact.target_path

    # get test data
    test_dataset = ds.dataset(source=artifact_latest_s3_path, format="parquet")

    import random

    testSize = 4
    test_dataset = test_dataset.to_table().to_pylist()
    start = random.randint(0, len(test_dataset) - testSize)
    test_dataset = test_dataset[start : start + testSize]

    for doc in test_dataset:
        document_id = doc["document_id"]
        contract_data = doc["text"]
        payload = json.dumps({"contract": contract_data})

        print(f"Testing {document_id}: {contract_data[:10]}")

        try:
            response = l_client.invoke(
                FunctionName="model_gateway",
                InvocationType="RequestResponse",
                Payload=payload,
            )
            print("✅ Invocation to lambda is sent successfully")
        except Exception as e:
            print("====> Invocation to lambda failed for some reason")
            print(e)

        if response:
            response_str = response["Payload"].read().decode("utf-8")
            hypotheses_list = json.loads(response_str)
            print("=======================================")
            print(hypotheses_list)
            print("=======================================")


def rolling_update_test_20mins(
    project,
    eval_data_key,
    eval_data_tag,
):

    import time

    interval = 5 * 60
    duration = 20 * 60

    start_time = time.monotonic()
    scheduled_times = [start_time + i * interval for i in range(duration // interval)]

    for scheduled_time in scheduled_times:

        # Define Lambda client with custom config
        custom_config = Config(
            region_name="us-east-1",
            read_timeout=900,
            connect_timeout=900,
            retries={"max_attempts": 1},
            max_pool_connections=10,
        )
        l_client = boto3.client("lambda", config=custom_config, region_name="us-east-1")

        now = time.monotonic()

        # Wait if we're early
        if now < scheduled_time:
            time.sleep(scheduled_time - now)
        try:
            test_lambda_few_rows(project, eval_data_key, eval_data_tag, l_client)
        except Exception as e:
            print(e)

    print("20 minute rolling test finished")
