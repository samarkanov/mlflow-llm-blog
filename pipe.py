import os
import time
import platform
import sys
import random
from datetime import datetime
import json
import mlflow
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

# --- LLM Configuration ---
JUDGE_MODEL = "gemini-2.5-flash"
LOCAL_MODEL = "google/gemma-3-1b-pt"
REMOTE_MODEL = "gemma-3-1b-it"
user_query = "Who is Bulgakov? Provide a brief summary of his life and key works."
max_tokens_to_generate = 128
num_generations_to_track = 1
system_prompt_template = "Answer the following question in two sentences: {query}"
MLFLOW_LOG_SYSTEM_METRICS = True

# mlflow
MLFLOW_TRACKING_URI="http://localhost:8082"

# Utility function to strip the prompt from the generated text
def strip_prompt_from_generation(prompt, generated_text):
    """
    Strips the original prompt from the beginning of the generated text,
    assuming the model echoes the prompt.
    """
    if generated_text.startswith(prompt):
        return generated_text[len(prompt):].strip()
    return generated_text.strip()


def _run_local_llm(ti):
    local_results = []

    local_results.append({
        'attempt': 1,
        'text': "aaa",
        'generation_time_seconds': 100
    })

    # Push the result to XCom
    ti.xcom_push(key='local_llm_result', value=local_results)


def _run_local_llm(ti):
    """
    Calls a local LLM model (Hugging Face pipeline) and pushes the result to XCom.
    Tracks generation details with MLflow.
    """
    from transformers import pipeline
    from huggingface_hub import login
    print("Starting _run_local_llm...")

    # --- Set MLflow Experiment ---
    # Moved inside the function to avoid top-level execution
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("My LLM Generation Experiments")
    mlflow.transformers.autolog() # Moved inside the function

    # --- Login to Hugging Face ---
    # Ensure HUGGINGFACE_TOKEN is set as an environment variable in Airflow worker pods
    try:
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if hf_token:
            login(token=hf_token)
        else:
            print("Warning: HUGGINGFACE_TOKEN environment variable not set. Local model download might fail for private models.")
    except Exception as e:
        print(f"Error logging into Hugging Face: {e}")

    # --- Start MLflow Run ---
    with mlflow.start_run(log_system_metrics=MLFLOW_LOG_SYSTEM_METRICS) as run:
        # --- Metadata & Tags ---
        mlflow.set_tag("task", "text_generation_series")
        mlflow.set_tag("model_family", "gemma_local")
        mlflow.set_tag("prompt_type", "factual_query")
        mlflow.set_tag("num_generation_attempts", num_generations_to_track)
        mlflow.set_tag("os_type", platform.system())
        mlflow.set_tag("os_release", platform.release())
        mlflow.set_tag("python_version", sys.version)
        mlflow.set_tag("python_executable", sys.executable)

        # --- Log Basic Params ---
        mlflow.log_param("local_model_name", LOCAL_MODEL)
        mlflow.log_param("user_query", user_query)
        mlflow.log_param("system_prompt_template", system_prompt_template)
        mlflow.log_param("max_tokens_to_generate", max_tokens_to_generate)

        # Construct the full prompt
        full_prompt = system_prompt_template.format(query=user_query)

        # --- Initialize Pipeline ---
        print(f"Initializing local pipeline with model: {LOCAL_MODEL}...")
        pipe_start = time.perf_counter()
        try:
            pipe = pipeline("text-generation", model=LOCAL_MODEL, torch_dtype="auto", device="cpu")
            # pipe = pipeline("text-generation", model=REMOTE_MODEL, torch_dtype="auto", device="cpu")
        except Exception as e:
            print(f"Error initializing Hugging Face pipeline: {e}")
            ti.xcom_push(key='local_llm_result', value={"error": str(e)})
            return # Exit early if pipeline fails to initialize

        pipe_end = time.perf_counter()
        init_time = pipe_end - pipe_start
        mlflow.log_metric("local_pipeline_init_time_seconds", init_time)
        print(f"Pipeline Initialization Time: {init_time:.4f} seconds\n")

        # --- Log Model to MLflow ---
        # Log the model with an example input/output
        dummy_input = "Hello, how are you?"
        dummy_output = pipe(dummy_input, max_new_tokens=10)[0]['generated_text']
        mlflow.transformers.log_model(
            transformers_model=pipe,
            artifact_path="local_llm_model",
            input_example=dummy_input,
            output_example=dummy_output
        )

        print(f"Input Query: '{user_query}'")
        print(f"Full Prompt sent to local LLM: '{full_prompt}'")

        # --- Run Generations ---
        local_results = []
        for i in range(1, num_generations_to_track + 1):
            print(f"--- Local LLM Generation Attempt {i}/{num_generations_to_track} ---")
            with mlflow.start_span(name=f"local_text_generation_attempt_{i}"):
                start = time.perf_counter()
                try:
                    output = pipe(full_prompt, max_new_tokens=max_tokens_to_generate, do_sample=True, temperature=0.7)
                    generated_text = output[0].get('generated_text', '')
                    # Strip the prompt from the generated text
                    clean_generated_text = strip_prompt_from_generation(full_prompt, generated_text)
                except Exception as e:
                    print(f"Error during local LLM generation attempt {i}: {e}")
                    clean_generated_text = f"Error generating text: {e}"

                end = time.perf_counter()
                gen_time = end - start

                local_results.append({
                    'attempt': i,
                    'text': clean_generated_text,
                    'generation_time_seconds': gen_time
                })

                mlflow.log_metric("local_gen_time_seconds_per_attempt", gen_time, step=i)
                mlflow.log_metric("local_generated_length_chars_per_attempt", len(clean_generated_text), step=i)
                mlflow.log_param(f"local_generated_output_attempt_{i}", clean_generated_text)

                print(f"Clean Generated Text (Local):\n{clean_generated_text}")
                print(f"Time Taken: {gen_time:.4f} seconds\n")

        print(f"MLflow Run ID for local LLM: {run.info.run_id}")
        print("Local LLM tracking complete.")

    # Push the result to XCom
    ti.xcom_push(key='local_llm_result', value=local_results)
    print(f"Local LLM operation completed.")


def _run_remote_llm(ti):
    """
    Calls a remote LLM model (Google Gemini) and pushes the result to XCom.
    Tracks generation details with MLflow.
    """
    import google.genai as genai
    print("Starting _run_remote_llm...")

    # --- Set MLflow Experiment ---
    # Moved inside the function to avoid top-level execution
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("My LLM Generation Experiments")
    mlflow.gemini.autolog() # Moved inside the function

    # Configure Gemini API
    google_api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
    if not google_api_key:
        print("Error: GOOGLE_GEMINI_API_KEY environment variable not set. Cannot run remote LLM.")
        ti.xcom_push(key='remote_llm_result', value={"error": "GOOGLE_GEMINI_API_KEY not set"})
        return

    # Moved inside the function to avoid top-level execution
    client = genai.Client(api_key=google_api_key)

    # --- Start MLflow Run ---
    with mlflow.start_run(log_system_metrics=MLFLOW_LOG_SYSTEM_METRICS) as run:
        # --- Metadata & Tags ---
        mlflow.set_tag("task", "text_generation_series")
        mlflow.set_tag("model_family", "gemini_remote")
        mlflow.set_tag("prompt_type", "factual_query")
        mlflow.set_tag("num_generation_attempts", num_generations_to_track)
        mlflow.set_tag("os_type", platform.system())
        mlflow.set_tag("os_release", platform.release())
        mlflow.set_tag("python_version", sys.version)
        mlflow.set_tag("python_executable", sys.executable)

        # --- Log Basic Params ---
        mlflow.log_param("remote_model_name", REMOTE_MODEL)
        mlflow.log_param("user_query", user_query)
        mlflow.log_param("system_prompt_template", system_prompt_template)
        mlflow.log_param("max_tokens_to_generate", max_tokens_to_generate)

        # Construct the full prompt
        full_prompt = system_prompt_template.format(query=user_query)

        # --- Run Generations ---
        remote_results = []
        for i in range(1, num_generations_to_track + 1):
            print(f"--- Remote LLM Generation Attempt {i}/{num_generations_to_track} ---")
            with mlflow.start_span(name=f"remote_text_generation_attempt_{i}"):
                start = time.perf_counter()
                generated_text = ""
                try:
                    # For Gemini, the prompt is typically passed directly to generate_content
                    response = client.models.generate_content(
                        model=REMOTE_MODEL,
                        contents=full_prompt
                    )
                    generated_text = response.text.strip()
                except Exception as e:
                    print(f"Error during remote LLM generation attempt {i}: {e}")
                    generated_text = f"Error generating text: {e}"

                end = time.perf_counter()
                gen_time = end - start

                remote_results.append({
                    'attempt': i,
                    'text': generated_text,
                    'generation_time_seconds': gen_time
                })

                mlflow.log_metric("remote_gen_time_seconds_per_attempt", gen_time, step=i)
                mlflow.log_metric("remote_generated_length_chars_per_attempt", len(generated_text), step=i)
                mlflow.log_param(f"remote_generated_output_attempt_{i}", generated_text)

                print(f"Generated Text (Remote):\n{generated_text}")
                print(f"Time Taken: {gen_time:.4f} seconds\n")

        print(f"MLflow Run ID for remote LLM: {run.info.run_id}")
        print("Remote LLM tracking complete.")

    # Push the result to XCom
    ti.xcom_push(key='remote_llm_result', value=remote_results)
    print(f"Remote LLM operation completed.")


def _select_best_answer(ti):
    """
    Pulls results from both parallel LLM operations and uses a JUDGE_MODEL
    to select the best answer.
    """
    import google.genai as genai
    print("Starting selection of the best answer...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Pull results from the two parallel tasks using their task_ids and XCom keys
    local_llm_results = ti.xcom_pull(task_ids='run_local_llm', key='local_llm_result') 
    remote_llm_results = ti.xcom_pull(task_ids='run_remote_llm', key='remote_llm_result')

    print(f"Results from local LLM: {local_llm_results}")
    print(f"Results from remote LLM: {remote_llm_results}")
    mlflow.log_param("results_from_local_llm", local_llm_results)
    mlflow.log_param("results_from_remote_llm", remote_llm_results)

    if not local_llm_results and not remote_llm_results:
        summary = "No results available from either LLM operation."
        print(summary)
        ti.xcom_push(key='final_best_result_summary', value=summary)
        return

    # Filter out potential error dictionaries
    local_answers = [r['text'] for r in local_llm_results if isinstance(r, dict) and 'text' in r] if local_llm_results else []
    remote_answers = [r['text'] for r in remote_llm_results if isinstance(r, dict) and 'text' in r] if remote_llm_results else []

    all_answers = []
    for i, ans in enumerate(local_answers):
        all_answers.append({'source': 'local', 'attempt': i + 1, 'text': ans})
    for i, ans in enumerate(remote_answers):
        all_answers.append({'source': 'remote', 'attempt': i + 1, 'text': ans})

    if not all_answers:
        summary = "No valid LLM generations to compare."
        print(summary)
        ti.xcom_push(key='final_best_result_summary', value=summary)
        return

    # --- Set MLflow Experiment ---
    # Moved inside the function to avoid top-level execution
    mlflow.set_experiment("My LLM Judging Experiments")
    mlflow.gemini.autolog() # Moved inside the function

    with mlflow.start_run(nested=True, log_system_metrics=MLFLOW_LOG_SYSTEM_METRICS) as run:
        mlflow.set_tag("task", "llm_judging")
        mlflow.set_tag("judge_model", JUDGE_MODEL)
        mlflow.log_param("original_query", user_query)
        mlflow.log_param("system_prompt_template", system_prompt_template)
        mlflow.log_param("num_answers_to_judge", len(all_answers))

        google_api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
        if not google_api_key:
            print("Error: GOOGLE_GEMINI_API_KEY environment variable not set. Cannot run judge LLM.")
            ti.xcom_push(key='final_best_result_summary', value={"error": "GOOGLE_GEMINI_API_KEY not set for judging"})
            return

        # Moved inside the function to avoid top-level execution
        client = genai.Client(api_key=google_api_key)
        print(f"Using judge model: {JUDGE_MODEL}")

        # Construct the judging prompt
        judging_instructions = (
            f"You are an expert evaluator of LLM responses. "
            f"The original user query was: '{user_query}'. "
            f"The original instruction for the LLMs was: '{system_prompt_template.format(query='')}'. "
            f"Evaluate the following answers for accuracy, relevance, and adherence to the 'two sentences' constraint. "
            f"Rank them from best to worst, providing a brief justification for each. "
            f"**IMPORTANT: Return your response ONLY as a JSON array of objects.** "
            f"Each object must have 'rank' (integer), 'source' (string), 'attempt' (integer), 'text' (string), and 'justification' (string). "
            f"Example format:\n"
            f"```json\n"
            f"[\n"
            f"  {{\n"
            f'    "rank": 1,\n'
            f'    "source": "local",\n'
            f'    "attempt": 1,\n'
            f'    "text": "Best answer text.",\n'
            f'    "justification": "This answer was accurate and concise."\n'
            f"  }}\n"
            f"]\n"
            f"```\n\n"
            f"Answers to evaluate:\n"
        )

        for i, answer in enumerate(all_answers):
            judging_instructions += f"--- Answer {i+1} (Source: {answer['source']}, Attempt: {answer['attempt']}) ---\n"
            judging_instructions += f"{answer['text']}\n\n"

        print("Sending judging request to LLM...")
        print(f"Judging instructions:\n{judging_instructions}")

        judge_response = client.models.generate_content(
            model=JUDGE_MODEL,
            contents=judging_instructions
        )

        # Try to parse the JSON response
        raw_judged_text = judge_response.text
        mlflow.log_param("judge_prompt", judging_instructions)
        mlflow.log_param("judge_raw_response", raw_judged_text)


with DAG(
    dag_id='llm_parallel_and_select_pipeline',
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['genai', 'llm', 'mlflow', 'comparison'],
    doc_md="""
    ### LLM Parallel Generation and Selection Airflow Pipeline
    This DAG demonstrates running two distinct LLM tasks (local Hugging Face and remote Gemini)
    in parallel. A subsequent task then uses a more advanced LLM (Gemini 2.5 Flash)
    to evaluate and select the "best" answer among the generated outputs.
    MLflow is extensively used for tracking all LLM generations and the judging process.
    """
) as dag:
    task_run_local_llm = PythonOperator(
        task_id='run_local_llm',
        python_callable=_run_local_llm,
        doc_md="""
        #### Parallel Task 1: Run Local LLM
        Initializes and calls a local Hugging Face model (`google/gemma-3-1b-pt`)
        multiple times to generate responses. Tracks metrics and parameters
        using MLflow, and pushes all generated texts to XCom.
        """
    )

    task_run_remote_llm = PythonOperator(
        task_id='run_remote_llm',
        python_callable=_run_remote_llm,
        doc_md="""
        #### Parallel Task 2: Run Remote LLM
        Initializes and calls a remote Google Gemini model (`gemini-1.5-flash`)
        multiple times to generate responses. Tracks metrics and parameters
        using MLflow (with auto-logging for Gemini), and pushes all generated texts to XCom.
        """
    )

    task_judge_and_select_best = PythonOperator(
        task_id='select_best_answer',
        python_callable=_select_best_answer,
        doc_md="""
        #### Select Best Answer Task
        Pulls all generated texts from `run_local_llm` and `run_remote_llm` via XComs.
        It then uses a more advanced LLM (`JUDGE_MODEL`, Gemini 2.5 Flash) to
        evaluate the quality, relevance, and adherence to instructions of each generated answer.
        The task selects and logs the "best" answer based on the judge's assessment.
        """
    )

    [task_run_local_llm, task_run_remote_llm] >> task_judge_and_select_best
