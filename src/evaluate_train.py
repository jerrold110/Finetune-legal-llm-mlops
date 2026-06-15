# ====== If run from notebooks, the working directory is /notebooks =====
import os, sys
parent_dir = os.path.abspath('..')
sys.path.append(parent_dir)

#print(sys.executable)
# ====== If run from notebooks, the working directory is /notebooks =====

import src.utils as utils #import utils

from datetime import datetime
############################################
import mlrun
from dotenv import load_dotenv
load_dotenv() 
from pathlib import Path

artifact_path = Path.cwd().parent
artifact_path = str(artifact_path.as_posix()) # convert windows path to unix path
artifact_path = "file://" + artifact_path
p = mlrun.set_environment("http://localhost:8080", artifact_path=artifact_path)
project = mlrun.load_project(name='finetune-legal-extractor', context="../") # project yaml must be in this directory
############################################



def evaluate_model_train(context,
                         train_dataset,
                         train_dataset_tag,
                         val_dataset,
                         val_dataset_tag,
                         test_dataset,
                         test_dataset_tag,
                         prompt,
                         prompt_tag,
                         ####
                         epochs,
                         batch_grad_accumulation,
                         learning_rate,
                         lora_r,
                         lora_alpha,
                         early_stopping_threshold
                         ):
    
    # Define key
    key = datetime.now().strftime("%Y%m%d_%H%M")

    # Preprocess train/eval data, process test data
    utils.prepare_train_datasets(
        project,
        train_dataset,
        train_dataset_tag,
        val_dataset,
        val_dataset_tag,
        test_dataset,
        test_dataset_tag,
        prompt,
        prompt_tag,
        key
    )

    # Train model and create loss graph
    commit_oid, log_data, s3_output_path = utils.train_model_get_outputs(
            key=key,
            model_repo="JerroldK/Hermes-4-14B-contract-extractor",
            model_revision="75875f970c359f89ad9e7d4dc86bf3c075c73c31",
            epochs=epochs,
            batch_grad_accumulation=batch_grad_accumulation,
            learning_rate=learning_rate,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            early_stopping_threshold=early_stopping_threshold,
        )
    print("Trainer log data:")
    print(log_data)

    # key, commit_oid = "20260603_1558", "39c89f599964a53e6dc2e11c273a6d2d6ad52a2e"
    # Evaluate model with HF commit oid
    dataset_metrics, s3_output_path_metric = utils.evaluate_model_lora(
        key,
        project,
        "JerroldK/Hermes-4-14B-contract-extractor", # model_repo
        "75875f970c359f89ad9e7d4dc86bf3c075c73c31", # model_revision
        prompt,
        prompt_tag,
        "JerroldK/H4-14b-contract-extractor-adapter", # adapter_repo
        commit_oid # adapter_revision
    )

    # Register results in the job run to MLRun
    keys = [
        "count",
        "average_accuracy",
        "average_fmeasure",
        "t_average_fmeasure",
        "t_average_perc_above_75fmeasure",
        "f_average_fmeasure",
        "f_average_perc_above_75fmeasure",
        "min_accuracy",
        "min_t_average_fmeasure",
        "min_t_perc_above_75fmeasure",
        "min_f_average_fmeasure",
        "min_f_perc_above_75fmeasure",
    ]

    for k in keys:
        context.log_result(k, dataset_metrics[k])

    print("Experiment logged")

    # anything you return will be accessible under RunObject.outputs()['return']
    return {"commit_oid:": commit_oid,
            "adapter_repo": "JerroldK/H4-14b-contract-extractor-adapter",
            "s3_output_path": s3_output_path}

if __name__ == "__main__":
    evaluate_model_train(train_dataset="raw-proc-process-raw_train_data",
                         train_dataset_tag="20260506_1224",
                         val_dataset="raw-proc-process-raw_validation_data",
                         val_dataset_tag="20260506_1224",
                         test_dataset="raw-proc-process-raw_test_data",
                         test_dataset_tag="20260506_1224",
                         prompt="contract_extractor_prompt",
                         prompt_tag="latest",
                         epochs=5,
                         batch_grad_accumulation=16,
                         learning_rate=2e-4,
                         lora_r=16,
                         lora_alpha=32,
                         early_stopping_threshold=1e-3)

