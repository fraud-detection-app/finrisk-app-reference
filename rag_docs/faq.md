# FAQ

Q: Why use PR-AUC instead of accuracy?
A: Fraud is rare, so a model can achieve high accuracy by predicting non-fraud. PR-AUC focuses on
positive-class ranking performance.

Q: What is the AWS deployment?
A: The project can upload curated data to S3, query it with Athena, run Streamlit on EC2 or locally,
and call Bedrock for LLM summaries.
