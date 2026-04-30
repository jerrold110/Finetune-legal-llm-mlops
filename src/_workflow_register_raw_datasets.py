"""
Assume that functions inside register_datset.py have been registered

Reference:
https://docs.mlrun.org/en/v1.7/projects/build-run-workflows-pipelines.html
"""
from kfp import dsl
import mlrun

@dsl.pipeline(
    name="register_raw_datasets",
    description="Something..."
)
def pipeline(source_path,
             version):
    """
    Registers 3 datasets: train, validate, test
    """
    # Train dataset
    runobj = mlrun.run_function("raw-proc",  # use the function name registered in the register_funcs.ipynb file
                inputs={"input_uri": f'{source_path}/train.parquet'},
                params={"label_column": "inference",
                        "artifact_key": f'train_data',
                        "version": version,
                        "output_uri_path": 's3://legal-llama-data/processed_training'},
                local=True
        )
    
    # Validation dataset
    runobj = mlrun.run_function("raw-proc", 
                inputs={"input_uri": f'{source_path}/validation.parquet'},
                params={"label_column": "inference",
                        "artifact_key": f'validation_data',
                        "version": version,
                        "output_uri_path": 's3://legal-llama-data/processed_training'},
                local=True
        )
    
    # Test dataset
    runobj = mlrun.run_function("raw-proc", 
                inputs={"input_uri": f'{source_path}/test.parquet'},
                params={"label_column": "inference",
                        "artifact_key": f'test_data',
                        "version": version,
                        "output_uri_path": 's3://legal-llama-data/processed_training'},
                local=True
        )
    
