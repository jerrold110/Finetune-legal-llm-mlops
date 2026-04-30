# ====== If run from notebooks, the working directory is /notebooks =====
import os, sys
parent_dir = os.path.abspath('..')
sys.path.append(parent_dir)
# ====== If run from notebooks, the working directory is /notebooks =====

import src.utils as utils
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

def evaluate_model(context,
                    dataset,
                    dataset_tag,
                    prompt,
                    prompt_tag):

    # Spin up endpoint
    print('Spinning up DJL endpoint')
    predictor, endpoint_name = utils.deploy_djl_contbat()
    print('Spun up DJL endpoint')
    #endpoint_name = "lmi-batch-Hermes-14B-FP8-2026-04-29-06-28-29-596"

    # Define prompt and datasets
    eval_data_key=dataset
    eval_data_tag=dataset_tag
    prompt_key=prompt
    prompt_tag=prompt_tag

    # Get all inferences and calculate metrics
    dataset_metrics, s3_output_path = utils.process_multiple_row_testdata(
        project,
        endpoint_name,
        eval_data_key,
        eval_data_tag,
        prompt_key,
        prompt_tag 
    )
    print('Inferences complete')

    # Save experiment metrics to MLRun
    keys = [
        "count",
        "average_accuracy",
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

    # Delete the endpoint
    predictor.delete_model()
    predictor.delete_endpoint()

    # anything you return will be accessible under RunObject.outputs()['return']
    return {"s3_output_path": s3_output_path}