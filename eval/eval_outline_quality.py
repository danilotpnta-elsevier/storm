"""Compute outline quality metrics on a dataset.

The script expects
    - a CSV file (args.input_path) with a column 'topic' containing the topics for evaluation.
    - a directory (args.gt_dir) containing human-written articles. The articles should be named as txt/{topic_name}.txt.
    - a directory (args.pred_dir) containing generated outlines. The outlines should be named as {topic_name}/{args.pred_file_name}.
"""

import os
import re
from argparse import ArgumentParser
import pandas as pd
from tqdm import tqdm
from metrics import heading_soft_recall, heading_entity_recall


def load_str(path):
    with open(path, "r") as f:
        return "\n".join(f.readlines())


def get_sections(path):
    s = load_str(path)
    s = re.sub(r"\d+\.\ ", "#", s)
    sections = []
    for line in s.split("\n"):
        line = line.strip()
        if "# References" in line:
            break
        if line.startswith("#"):
            if any(
                keyword in line.lower()
                for keyword in ["references", "external links", "see also", "notes"]
            ):
                break
            sections.append(line.strip("#").strip())
    return sections


def main(args):
    df = pd.read_csv(args.input_path)
    entity_recalls = []
    heading_soft_recalls = []
    topics = []
    skipped_topics = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        topic_name = row["topic"].replace(" ", "_").replace("/", "_")

        # Handle missing ground truth or prediction files
        try:
            gt_path = os.path.join(args.gt_dir, "txt", f"{topic_name}.txt")
            pred_path = os.path.join(args.pred_dir, topic_name, args.pred_file_name)

            if not os.path.exists(gt_path):
                raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
            if not os.path.exists(pred_path):
                raise FileNotFoundError(f"Prediction file not found: {pred_path}")

            gt_sections = get_sections(gt_path)
            pred_sections = get_sections(pred_path)

            entity_recalls.append(
                heading_entity_recall(
                    golden_headings=gt_sections, predicted_headings=pred_sections
                )
            )
            heading_soft_recalls.append(heading_soft_recall(gt_sections, pred_sections))
            topics.append(row["topic"])

        except FileNotFoundError as e:
            print(f"Warning: {e}")
            skipped_topics.append(row["topic"])
            continue

    results = pd.DataFrame(
        {
            "topic": topics,
            "entity_recall": entity_recalls,
            "heading_soft_recall": heading_soft_recalls,
        }
    )
    results_output_path = os.path.join(
        args.result_output_dir, "storm_outline_quality.csv"
    )
    results.to_csv(results_output_path, index=False)

    # Print averages
    if entity_recalls and heading_soft_recalls:
        avg_entity_recall = sum(entity_recalls) / len(entity_recalls)
        avg_heading_soft_recall = sum(heading_soft_recalls) / len(heading_soft_recalls)
        print(f"Average Entity Recall: {avg_entity_recall}")
        print(f"Average Heading Soft Recall: {avg_heading_soft_recall}")
    else:
        print("No valid topics processed.")

    if skipped_topics:
        print("\nSkipped topics:")
        for topic in skipped_topics:
            print(f"- {topic}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--input-path",
        type=str,
        help="Path to the CSV file storing topics and ground truth URLs.",
    )
    parser.add_argument("--gt-dir", type=str, help="Path of human-written articles.")
    parser.add_argument("--pred-dir", type=str, help="Path of generated outlines.")
    parser.add_argument(
        "--pred-file-name",
        default="storm_gen_outline.txt",
        type=str,
        help="Name of the outline file.",
    )
    parser.add_argument(
        "--result-output-dir", help="Directory to store the evaluation results. "
    )
    args = parser.parse_args()

    if not os.path.exists(args.result_output_dir):
        os.makedirs(args.result_output_dir)
        print(f"Directory {args.result_output_dir} created.")

    """
    python eval_outline_quality.py \
        --input-path "../TopicPagesWiki/topics_ores_scores.csv" \
        --gt-dir "../TopicPagesWiki" \
        --pred-dir "/home/toapantabarahonad/ds-agentic-topic-pages-gen/data/baseline/refined_articles/models--snippet_ranking_model" \
        --result-output-dir "/home/toapantabarahonad/ds-agentic-topic-pages-gen/results/storm_outline_eval_results"
    """

    main(args)
