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

def deploy_new_adapter(
    context,
    adapter_revision,
    endpoint_name,
    base_ic_name,
    test_dataset,
    test_dataset_tag,
    prompt,
    prompt_tag,
    rolling_update:bool,
    deployment_color:str
    ):

    assert deployment_color in ("Black", "White")

    # Define key
    key = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Deploy LORA adapter as IC adapter
    print("======> Beginning adapter deployment")
    adapter, adapter_name = utils.deploy_sm_lora_adapter(
        key,
        endpoint_name,
        base_ic_name,
        adapter_revision
    )

    # adapter_name = "adapter-lmi-Hermes-FP8-2026-07-16-07-42-31-222"
    print("Logs are under: /aws/sagemaker/InferenceComponents/base-lmi-Hermes-FP8-2026-xxxxxx")

    # Run load tests on new model/adapter
    print("======> Beginning load test")
    dataset_metrics, s3_eval_path = utils.process_multiple_row_testdata(
        project,
        endpoint_name,
        adapter_name,
        test_dataset,
        test_dataset_tag,
        prompt,
        prompt_tag,
        key
    )
    
    check_count = dataset_metrics['count'] >= 2 # in real this would be 100
    check_accuracy = dataset_metrics['average_accuracy'] >= 0.7
    check_fmeasure = dataset_metrics['average_fmeasure'] >= 0.7

    result = check_count and check_accuracy and check_fmeasure
    if not result:
        er = f"Load test failed with values/n check_count: {check_count}\n check_accuracy: {check_accuracy}\n check_fmeasure {check_fmeasure}"
        raise Exception(er)
    
    print(f"✅ Endpoint passed load test with metrics:\n {dataset_metrics}")

    adapter_name = 'adapter-lmi-Hermes-FP8-2026-07-21-09-43-54-983'
    endpoint_name = 'lmi-Hermes-FP8-2026-07-21-06-34-06-259'

    # Gradually shift traffic to new model by sending request to AppConfig with pre-configured deployment policy with rollback
    print("Sending request to AppConfig to change variables in Lambda")
    utils.update_gateway_destination_sm(
        model_endpoint=endpoint_name,
        model_adapter=adapter_name,
        template_uri="s3://legal-llama-data/llm_prompt/contract_extractor_prompt/20260610_1257/contract_extractor_prompt.json",
        rolling=rolling_update,
        deployment_color=deployment_color
    )

    if rolling_update:
        print("✅ Deployment workflow finished, rolling update in progress, now monitoring new model in preparation for rollback")

        # Test to simulate production workloads during rolling update
        utils.rolling_update_test_20mins(
            project,
            test_dataset,
            test_dataset_tag,
        )

    else:
        print("✅ Direct deployment finished, now monitoring new model in preparation for appconfig rollback for remaining duratino of bake time (if it exists)")

    return {
        'endpoint_name': endpoint_name,
        'base_ic_name': base_ic_name,
        'adapter_name': adapter_name
    }

