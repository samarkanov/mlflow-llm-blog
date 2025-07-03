all:


rescan:
	cp pipe.py ~/airflow/dags/
	python3 -c "from tools import pipelineRescanTool; pipelineRescanTool()"

start-mlflow:
	mlflow server --host localhost --port 8082

start-airflow:
	airflow standalone
