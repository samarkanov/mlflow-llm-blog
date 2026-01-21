Source code for the blog entry: [https://samarkanov.info/blog/2025/jun/local-remote-llms-airflow-mlflow.html](https://samarkanov.info/blog/2025/jun/local-remote-llms-airflow-mlflow.html)

I'm developing an Airflow pipeline that benchmarks local LLM performance against remote models by running concurrent inferences and using a high-power 'judge' LLM to select the superior response. Using MLflow 3.1.0 to track metrics and capture outputs, I’ve concluded that local execution remains a hardware-intensive and time-consuming 'painful endeavor' for standard consumer computers in 2025.

![](https://samarkanov.info/assets/llm-mlflow-airflow/llm-airflow-mlflow.svg)
