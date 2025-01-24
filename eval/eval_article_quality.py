"""Compute article quality metrics on a dataset.

The script expects
    - a CSV file (args.input_path) with a column 'topic' containing the topics for evaluation.
    - a directory (args.gt_dir) containing human-written articles. The articles should be named as txt/{topic_name}.txt
        and there should be a json file named json/{topic_name}.json containing the named entities in the article.
    - a directory (args.pred_dir) containing generated articles. The outlines should be named as {topic_name}/{args.pred_file_name}.
"""

import os
import logging
import argparse

import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import torch
from transformers import AutoTokenizer, LlamaForCausalLM

from evaluation_prometheus import get_grading_dict, preprocess_text
from evaluation_trim_length import process_document
from metrics import article_entity_recall, compute_rouge_scores

from src.utils import dump_json, load_json, load_str
from config.constants import HF_CACHE_DIR

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ColoredFormatter(logging.Formatter):
    COLORS = {
        "WARNING": "\033[93m",  # Yellow
        "INFO": "\033[97m",  # White
        "DEBUG": "\033[92m",  # Green
        "CRITICAL": "\033[94m",  # Blue
        "ERROR": "\033[91m",  # Red
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        record.levelname = color + record.levelname + self.COLORS["RESET"]
        record.msg = color + str(record.msg) + self.COLORS["RESET"]
        return super().format(record)


def check_files_exists(file_path_1, file_path_2, file_path_3):
    """
    Check if all necessary files exist. Return False if any are missing. Otherwise, return True.
    Log a warning for missing files and their paths.

    Example:
        - pred_article_path:    Boltzmann_Distribution/storm_gen_article_polished.txt
        - gt_article_path:      TopicPagesWiki/txt/Boltzmann_Distribution.txt
        - gt_article_json_path: TopicPagesWiki/json/Boltzmann_Distribution.json
    """
    file_paths = {
        "Prediction": file_path_1,
        "Ground Truth (txt)": file_path_2,
        "Ground Truth (json)": file_path_3,
    }
    missing_files = {
        desc: path for desc, path in file_paths.items() if not os.path.exists(path)
    }

    if missing_files:
        for desc, path in missing_files.items():
            logger.warning(f"{desc} file not found: {path}")
        return False
    return True


def update_aggregated_results(aggregated_results, evaluation_main_dict):
    """
    Update aggregated results with evaluation results from one topic.

    Args:
        aggregated_results (dict): Dictionary containing aggregated results.
        evaluation_main_dict (dict): Dictionary containing evaluation results for one topic.
    """

    for k, v in evaluation_main_dict["grading"]["rubric_grading"].items():
        aggregated_results[k].append(v)

    for k, v in evaluation_main_dict["grading"]["auto_grading"].items():
        aggregated_results[k].append(v)

    aggregated_results["entity_recall"].append(
        evaluation_main_dict["grading"]["entity_recall"]
    )


def compute_average_scores(aggregated_results):
    """
    Compute average scores from aggregated results and save to a JSON file.

    Args:
        aggregated_results (dict): Dictionary containing aggregated results.

    Returns:
        avg_results (dict): Dictionary containing average scores.
    """
    logger.info(f"Computing average score.")
    avg_results = {}

    # for k in aggregated_results:
    #     if type(aggregated_results[k][0]) is dict:
    #         avg_results[k] = sum(
    #             [float(x["score"]) for x in aggregated_results[k]]
    #         ) / len(aggregated_results[k])
    #     else:
    #         avg_results[k] = sum(aggregated_results[k]) / len(aggregated_results[k])
    #     print(f"{k}: {avg_results[k]}")

    # TODO: Revisist cases where scores is not present
    for k, v in aggregated_results.items():
        try:
            scores = []
            if isinstance(v, list):
                for entry in v:
                    if isinstance(entry, dict):
                        score = entry.get("score")
                        if isinstance(score, (int, float)):
                            scores.append(float(score))
                        elif (
                            isinstance(score, str)
                            and score.replace(".", "", 1).isdigit()
                        ):
                            scores.append(float(score))
                    elif isinstance(entry, (int, float)):
                        scores.append(float(entry))

            avg_results[k] = sum(scores) / len(scores) if scores else None
            logger.info(f"{k}: {avg_results[k]}")

        except Exception as e:
            logger.error(f"Error processing key '{k}': {e}")
            avg_results[k] = None

    return avg_results


def main(args):

    logger.info(f"loading tokenizer {args.tokenizer} and model {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = LlamaForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.float16,
        offload_folder=args.offload_dir,
        # max_memory={
        #     0: "23GiB",
        #     "cpu": "128GiB",
        # }
    )

    df = pd.read_csv(args.input_path)
    aggregated_results = defaultdict(list)

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing topics"):

        if i == 3:
            break
        topic = row["topic"]
        topic_name = topic.replace(" ", "_").replace("/", "_")
        pred_article_path = os.path.join(args.pred_dir, topic_name, args.pred_file_name)
        gt_article_path = os.path.join(args.gt_dir, "txt", topic_name + ".txt")
        gt_article_json_path = os.path.join(args.gt_dir, "json", topic_name + ".json")

        files_present = check_files_exists(
            pred_article_path, gt_article_path, gt_article_json_path
        )
        if not files_present:
            logger.warning(f"Skipping topic: {topic_name}")
            continue

        # Load files
        golden_answer_json = load_json(gt_article_json_path)
        golden_answer_txt = load_str(gt_article_path)
        golden_answer = preprocess_text(golden_answer_txt)

        pred_article_txt = load_str(pred_article_path)
        pred_article = preprocess_text(pred_article_txt)

        # Prometheus model has a limited context window.
        trimmed_output_for_rubric_grading = process_document(
            pred_article_path, max_words=2000
        )

        ### Computing evaluation metrics
        logger.info(f"Processing rubric grading.")
        evaluation_main_dict = {"topic": topic, "grading": {}}

        # 1. Get `rubric_grading` using Prometheus model
        grading_dict = get_grading_dict(
            responses=[trimmed_output_for_rubric_grading],
            topic=topic,
            tokenizer=tokenizer,
            model=model,
            prompt_template_path=args.prompt_template_path,
            rubric_path=args.rubric_path,
            logger=logger,
        )

        for criteria_description, response_grading_dict in grading_dict.items():
            for response_idx, feedback_dict in response_grading_dict.items():
                if "rubric_grading" not in evaluation_main_dict["grading"]:
                    evaluation_main_dict["grading"] = {
                        "rubric_grading": {criteria_description: feedback_dict}
                    }
                else:
                    evaluation_main_dict["grading"]["rubric_grading"][
                        criteria_description
                    ] = feedback_dict

        # 2. Get `auto_grading` ~ automatic evaluation scores
        logger.info(f"Processing automatic evaluation.")
        automatic_evaluation_score = compute_rouge_scores(
            predicted_answer=pred_article, golden_answer=golden_answer
        )
        evaluation_main_dict["grading"]["auto_grading"] = automatic_evaluation_score

        # 3. Get `entity_recall` ~ named entity overlap with golden answer
        logger.info(f"Processing entity overlap with ground truth")
        evaluation_main_dict["grading"]["entity_recall"] = article_entity_recall(
            golden_entities=golden_answer_json["flair_entities"],
            predicted_article=pred_article,
        )
        ###

        # Save evaluation results
        results_one_topic_path = os.path.join(
            args.result_output_dir, f"{topic_name}.json"
        )
        dump_json(evaluation_main_dict, results_one_topic_path)

        # Store evaluation results in aggregated_results
        update_aggregated_results(aggregated_results, evaluation_main_dict)

    ### End of loop
    results_all_topics_path = os.path.join(
        args.result_output_dir, "aggregated_results.json"
    )
    dump_json(aggregated_results, results_all_topics_path)
    # aggregated_results = load_json(results_all_topics_path)
    
    
    avg_results = compute_average_scores(aggregated_results)
    avg_results_path = os.path.join(args.result_output_dir, "avg_results.json")
    dump_json(avg_results, avg_results_path)


if __name__ == "__main__":
    # configure logger
    global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    formatter = ColoredFormatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # command line argument
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=str,
        help="Using csv file to store topic and ground truth url at present.",
    )
    parser.add_argument(
        "--pred-dir", help="Directory to the file containing the LLM output."
    )
    parser.add_argument(
        "--gt-dir", help="Directory to the file containing the human-written articles."
    )
    parser.add_argument(
        "--result-output-dir",
        help="Directory to store the evaluation results. "
        "Each article evaluation will be saved as separate file named after {topic_name}.json",
    )
    parser.add_argument(
        "--pred-file-name",
        default="storm_gen_article_polished.txt",
        help="Name of the article file.",
    )
    parser.add_argument(
        "--prompt-template-path",
        default="./eval_prometheus_no_ref.prompt",
        help="path to evaluation prometheus prompt template",
    )
    parser.add_argument(
        "--rubric-path", default="./eval_rubric_5.json", help="path to rubric json file"
    )

    parser.add_argument("--tokenizer", default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument(
        "--model",
        choices=["kaist-ai/prometheus-13b-v1.0", "kaist-ai/prometheus-7b-v1.0"],
        default="kaist-ai/prometheus-13b-v1.0",
        # default="kaist-ai/prometheus-7b-v1.0",
        help="Model to use for rubric evaluation.",
    )
    parser.add_argument(
        "--offload_dir",
        default=os.path.join(HF_CACHE_DIR, "offload"),
        help="Directory to offload the model and tokenizer to.",
    )
    args = parser.parse_args()

    model_name = args.model.split("/")[-1]
    args.result_output_dir = os.path.join(args.result_output_dir, model_name)
    
    if not os.path.exists(args.result_output_dir):
        os.makedirs(args.result_output_dir)
        logger.info(f"Directory {args.result_output_dir} created.")

    if not os.path.exists(args.offload_dir):
        os.makedirs(args.offload_dir)
        logger.info(f"Directory {args.offload_dir} created.")

    """
    python eval_article_quality.py \
            --input-path "../TopicPagesWiki/topics_ores_scores.csv" \
            --gt-dir "../TopicPagesWiki" \
            --pred-dir "/home/toapantabarahonad/storm-plus/data/baseline/refined_articles/models--snippet_ranking_model" \
            --result-output-dir "/home/toapantabarahonad/storm-plus/results/storm_article_eval_results"
    """
    main(args)
