"""
Assume that functions inside register_datset.py have been registered

Reference:
https://docs.mlrun.org/en/v1.7/projects/build-run-workflows-pipelines.html
"""
from kfp import dsl
import mlrun

@dsl.pipeline(
    name="register_datasets",
    description="Something..."
)
def pipeline(#project: mlrun.projects.project.MlrunProject,
             source_path,
             version):
    """
    Registers 3 datasets: train, validate, test
    """
    # Train dataset
    runobj = mlrun.run_function("data-register-function", 
                inputs={"source_url": f'{source_path}/train.jsonl'},
                params={"label_column": "inference",
                        "artifact_key": f'train_data:{version}',
                        "version": version},
                local=True
        )
    
    # Validation dataset
    runobj = mlrun.run_function("data-register-function", 
                inputs={"source_url": f'{source_path}/validation.jsonl'},
                params={"label_column": "inference",
                        "artifact_key": f'validation_data:{version}',
                        "version": version},
                local=True
        )
    
    # Test dataset
    runobj = mlrun.run_function("data-register-function", 
                inputs={"source_url": f'{source_path}/test.jsonl'},
                params={"label_column": "inference",
                        "artifact_key": f'test_data:{version}',
                        "version": version},
                local=True
        )
    
