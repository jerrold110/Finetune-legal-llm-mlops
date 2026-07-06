# ====== If run from notebooks, the working directory is /notebooks =====
import os, sys
parent_dir = os.path.abspath('..')
sys.path.append(parent_dir)
# ====== If run from notebooks, the working directory is /notebooks =====

import src.utils as utils
from datetime import datetime
############################################
import mlrun
# Loads AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY and MLRUN_AWS_ROLE_ARN
from dotenv import load_dotenv
load_dotenv() 
from pathlib import Path

artifact_path = Path.cwd().parent
artifact_path = str(artifact_path.as_posix()) # convert windows path to unix path
artifact_path = "file://" + artifact_path
p = mlrun.set_environment("http://localhost:8080", artifact_path=artifact_path)
project = mlrun.load_project(name='finetune-legal-extractor', context="../") # project yaml must be in this directory
############################################

def deploy_new_model_adapter(
    context,
    adapter_revision,
    test_dataset,
    test_dataset_tag,
    prompt,
    prompt_tag,
    rolling_update:bool 
    ):

    # Define key
    key = datetime.now().strftime("%Y%m%d_%H%M")

    # Deploy model with IC component
    endpoint_name, base_ic_name = utils.deploy_lora_model()

    # Deploy adapter as IC adapter
    adapter, adapter_name = utils.deploy_lora_adapter(
        key,
        endpoint_name,
        base_ic_name,
        adapter_revision
    )

    # # Run load tests
    # dataset_metrics, s3_eval_path = utils.process_multiple_row_testdata(
    #     project,
    #     endpoint_name,
    #     adapter_name,
    #     test_dataset,
    #     test_dataset_tag,
    #     prompt,
    #     prompt_tag,
    #     key
    # )
    
    # print(f"✅ Endpoint passed load test with metrics:\n {dataset_metrics}")
    # check_count = dataset_metrics['count'] >= 100
    # check_accuracy = dataset_metrics['average_accuracy'] >= 0.8
    # check_fmeasure = dataset_metrics['average_fmeasure'] >= 0.8

    # result = check_count and check_accuracy and check_fmeasure

    # # If tests pass, gradually shift traffic to new model by sending request to AppConfig with pre-configured deployment policy
    # if result:
    #     print("Sending request to AppConfig to change variables in Lambda")
    #     pass
    # else:
    #     raise ValueError(f"The endpoint smoke test failed with values: {dataset_metrics}")
    
    return {
        'endpoint_name': base_ic_name,
        'adapter_name': adapter_name
    }

