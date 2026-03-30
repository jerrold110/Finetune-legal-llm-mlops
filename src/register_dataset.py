import mlrun

def register_processed_data(context, artifact_key, label_column, source_url: mlrun.DataItem, version:str):
    """
    When you pass your S3 path via the inputs={"source_url": source_url} dictionary in your .run() command, MLRun intercepts that string and automatically converts it into a powerful mlrun.DataItem object before handing it to your prep_data function.
    """
    # print(type(context)) # <class 'mlrun.execution.MLClientCtx'>

    # source_url is an s3 identifier to a jsonl file
    df = source_url.as_df(format="json", lines=True)

    # permitted formats csv|parquet|pq|tsdb|kv
    # https://docs.mlrun.org/en/latest/api/mlrun.execution/index.html#mlrun.execution.MLClientCtx.log_dataset

    context.log_dataset(key=artifact_key, 
                        df=df, 
                        index=False,
                        label_column=label_column,
                        artifact_path=f's3://legal-llama-data/registered/', 
                        format="csv", 
                        labels={'version':version},
                        upload=True)