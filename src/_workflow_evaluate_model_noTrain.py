"""
Assume that functions inside register_datset.py have been registered

Reference:
https://docs.mlrun.org/en/v1.7/projects/build-run-workflows-pipelines.html
"""
from kfp import dsl
import mlrun

@dsl.pipeline(
    name="evaluate_model_noTrain",
    description="abcd"
)
def pipeline(hf_url,
             hf_tag,
             dataset,
             dataset_tag,
             prompt,
             prompt_tag):
    """
    # Spin up an endpoint then
    # Evaluate the model with the dataset and prompt
    # Then place the results in S3
    # Log the experiment
    """
    params = {
        'hf_url': hf_url,
        'hf_tag': hf_tag,
        'dataset': dataset,
        'dataset_tag': dataset_tag,
        'prompt': prompt,
        'prompt_tag': prompt_tag,
    }
    
    runobj = mlrun.run_function(
        function="re_f1",  # use the function name registered in the register_funcs.ipynb file
        params=params,
        local=True)
    
    print('done...')
    
    
