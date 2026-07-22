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
    rolling_update:bool,
    deployment_color:str
    ):

    assert deployment_color in ("Black", "White")

    # Define key
    key = datetime.now().strftime("%Y%m%d_%H%M")

    # Deploy model with base component for LoRA adapter
    endpoint_name, base_ic_name = utils.deploy_sm_lora_model(
        instance="ml.g6e.2xlarge"
        #batch_size="7",
        #batch_tokens="30000" # throughput vs latency. Bottleneck is either memory (too high) or prefill (too low). 16000-40000
        # 40000, 10 failed completely. Prefill bottleneck.
        # 28600/31000, 8 is close, 1 fail. Decode bottleneck, not enough memory
        # Next to try is batch size 7 with 28600(7150*4, 7150*5), to maximise throughput during decoding. Try to learn how to optimise memory.

        # 31000, 7x5, fp8, 0.95. 4 failed out of 35. Decode bottleneck. 
        # Reduce to 5000*5+1000~26000. 5 prefill at once, more memory for kvcache-decoding. This caused 6 out of 7 to fail. Prefill bottleneck
        # Best for 7 is 30000/31000
    )

    # Deploy LORA adapter as IC adapter
    adapter, adapter_name = utils.deploy_sm_lora_adapter(
        key,
        endpoint_name,
        base_ic_name,
        adapter_revision
    )

    # key = "20260714_1748"

    # endpoint_name, adapter_name = "lmi-Hermes-FP8-2026-07-14-09-48-32-497","adapter-lmi-Hermes-FP8-2026-07-14-09-48-32-497"
    # print("Logs are under: /aws/sagemaker/InferenceComponents/base-lmi-Hermes-FP8-2026-xxxxxx")

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
    
    check_count = dataset_metrics['count'] >= 1 # in real this would be 100
    check_accuracy = dataset_metrics['average_accuracy'] >= 0.7
    check_fmeasure = dataset_metrics['average_fmeasure'] >= 0.7

    result = check_count and check_accuracy and check_fmeasure
    if not result:
        raise Exception(f"Load test failed with values/n check_count: {dataset_metrics['count']}\n check_accuracy: {dataset_metrics['average_accuracy']}\n check_fmeasure {dataset_metrics['average_fmeasure']}")
    print(f"✅ Endpoint passed load test with metrics:\n {dataset_metrics}")
    # endpoint_name = "lmi-Hermes-FP8-2026-07-21-06-34-06-259"
    # adapter_name = "adapter-lmi-Hermes-FP8-2026-07-21-06-49-32-003"
    # base_ic_name = "base-lmi-Hermes-FP8-2026-07-21-06-49-32-003"

    # LAST OF ALL: Gradually shift traffic to new model by sending request to AppConfig with pre-configured deployment policy with rollback
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
    else:
        print("✅ Direct deployment finished, now monitoring new model in preparation for rollback (if it exists)")

    return {
        'endpoint_name': endpoint_name,
        'base_ic_name': base_ic_name,
        'adapter_name': adapter_name
    }

