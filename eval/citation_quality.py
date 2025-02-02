"""The code is adapted from https://github.com/princeton-nlp/ALCE/blob/main/eval.py"""

import os
import re
import copy
import logging
import random
from tqdm import tqdm
from argparse import ArgumentParser

import numpy as np
import pandas as pd

import nltk
from nltk import sent_tokenize

from vllm import LLM, SamplingParams
from src.utils import dump_json, load_json, load_str, format_args
from config.constants import TOPICS_PER_CATEOGORY_JSON

random.seed(0)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)


def remove_citations(s):
    """Remove citations from a string."""

    return re.sub(r"\[\d+\]", "", s)


def truncate_paragraph(paragraph, max_words):
    # Tokenize paragraph into sentences
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", paragraph)

    # Tokenize each sentence into words and form trunks
    trunks = []
    current_trunk = []
    current_word_count = 0

    for sentence in sentences:
        sentence_words = sentence.split()  # Tokenize sentence into words
        sentence_word_count = len(sentence_words)

        if current_word_count + sentence_word_count <= max_words:
            current_trunk.append(sentence)
            current_word_count += sentence_word_count
        else:
            trunks.append(" ".join(current_trunk))
            current_trunk = [sentence]
            current_word_count = sentence_word_count

    if current_trunk:
        trunks.append(" ".join(current_trunk))

    return trunks


def process_citation_quality(citation_data):

    categories = load_json(TOPICS_PER_CATEOGORY_JSON)
    metrics = {
        topic: {"recall": recall, "precision": precision}
        for topic, recall, precision in zip(
            citation_data["topic"], citation_data["recall"], citation_data["precision"]
        )
    }

    results = {}
    for category, topics in categories.items():
        topic_metrics = [metrics[topic] for topic in topics if topic in metrics]
        recalls = [metric["recall"] for metric in topic_metrics]
        precisions = [metric["precision"] for metric in topic_metrics]

        results[category] = {
            "topics": [
                {"topic": topic, **metrics[topic]}
                for topic in topics
                if topic in metrics
            ],
            "stats": {
                "count": len(topic_metrics),
                "recall_mean": np.mean(recalls) if recalls else 0,
                "recall_std": np.std(recalls) if recalls else 0,
                "precision_mean": np.mean(precisions) if precisions else 0,
                "precision_std": np.std(precisions) if precisions else 0,
            },
        }
    return results


def _run_nli_autoais(passage, claim, partial):
    """
    Run inference for assessing AIS between a premise and hypothesis using vLLM.
    Adapted from the original AutoAIS implementation to work with vLLM.

    Args:
        passage (str): The source text passage
        claim (str): The claim to verify
        partial (bool): Whether to check for partial support

    Returns:
        int: 1 if the claim is supported, 0 otherwise
    """

    passage_trunks = truncate_paragraph(passage, 500)
    inference = 0

    for trunk in passage_trunks:
        # Construct the prompt
        if partial:
            prompt = f"Can the source at least partially support the claim? Start your answer with 'Yes' or 'No'.\nSource: {trunk}\nClaim: {claim}"
        else:
            prompt = f"Is the claim faithful to the source? A claim is faithful to the source if the core part in the claim can be supported by the source.\nStart your answer with 'Yes' or 'No'.\nSource: {trunk}\nClaim: {claim}"

        # Create the message in Mistral chat format
        messages = [{"role": "user", "content": prompt}]

        # Generate using vLLM
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=200,
        )
        # vLLM uses a different generation API
        outputs = mistral_7b_instruct_vllm.chat(messages, sampling_params)

        # Extract the generated text from the output
        generated_text = outputs[0].outputs[0].text.strip()
        print("response_vllm: ", generated_text)

        # Check if the response starts with "Yes"
        if generated_text.startswith("Yes"):
            inference = 1
            print("inference_vllm: ", inference)
            break
    print("inference_vllm: ", inference)
    return inference


def compute_autoais(
    data, decontext=False, concat=False, qampari=False, at_most_citations=None
):
    """
    Compute AutoAIS score.

    Args:
        data: requires field `output` and `docs`
              - docs should be a list of items with fields `title` and `text` (or `phrase` and `sent` for QA-extracted docs)
        citation: check citations and use the corresponding references.
        decontext: decontextualize the output
    """

    def _format_document(doc):
        """Format document for AutoAIS."""

        if "sent" in doc:
            # QA-extracted docs
            return "Title: %s\n%s" % (doc["title"], doc["sent"])
        else:
            return "Title: %s\n%s" % (doc["title"], doc["text"])

    ais_scores = []
    ais_scores_prec = []

    sent_total = 0
    sent_mcite = 0
    sent_mcite_support = 0
    sent_mcite_overcite = 0

    eval_log = []

    for item in tqdm(data):
        # Get sentences by using NLTK
        if qampari:
            sents = [
                item["question"] + " " + x.strip()
                for x in item["output"].rstrip().rstrip(".").rstrip(",").split(",")
            ]
        else:
            sents = sent_tokenize(item["output"])
        if len(sents) == 0:
            continue

        target_sents = [remove_citations(sent).strip() for sent in sents]

        entail = 0
        entail_prec = 0
        total_citations = 0
        for sent_id, sent in enumerate(sents):
            target_sent = target_sents[
                sent_id
            ]  # Citation removed and (if opted for) decontextualized
            joint_entail = -1  # Undecided

            # Find references
            ref = [
                int(r[1:]) - 1 for r in re.findall(r"\[\d+", sent)
            ]  # In text citation id starts from 1
            logger.info(f"For `{sent}`, find citations {ref}")
            if len(ref) == 0:
                # No citations
                joint_entail = 0
            elif any([ref_id >= len(item["docs"]) for ref_id in ref]):
                # Citations out of range
                joint_entail = 0
            else:
                if at_most_citations is not None:
                    ref = ref[:at_most_citations]
                total_citations += len(ref)
                joint_passage = "\n".join(
                    [_format_document(item["docs"][psgs_id]) for psgs_id in ref]
                )

            # If not directly rejected by citation format error, calculate the recall score
            if joint_entail == -1:
                # joint_entail = _run_nli_autoais(
                #     joint_passage, target_sent, partial=False
                # )
                joint_entail = _run_nli_autoais(
                    joint_passage, target_sent, partial=False
                )

            entail += joint_entail
            if joint_entail == 0:
                logger.info(f"[Unsupported sentence] {sent}")
            if len(ref) > 1:
                sent_mcite += 1

            unnecessary_citations = []

            # calculate the precision score if applicable
            if joint_entail and len(ref) > 1:
                sent_mcite_support += 1
                # Precision check: did the model cite any unnecessary documents?
                for psgs_id in ref:
                    # condition A
                    passage = _format_document(item["docs"][psgs_id])
                    # nli_result = _run_nli_autoais(passage, target_sent, partial=True)
                    nli_result = _run_nli_autoais(passage, target_sent, partial=True)

                    # condition B
                    if not nli_result:
                        subset_exclude = copy.deepcopy(ref)
                        subset_exclude.remove(psgs_id)
                        passage = "\n".join(
                            [
                                _format_document(item["docs"][pid])
                                for pid in subset_exclude
                            ]
                        )
                        # nli_result = _run_nli_autoais(
                        #     passage, target_sent, partial=False
                        # )
                        nli_result = _run_nli_autoais(
                            passage, target_sent, partial=False
                        )
                        if nli_result:  # psgs_id is not necessary
                            flag = 0
                            sent_mcite_overcite += 1
                            logger.info(
                                f"[Unnecessary citation] sent: {sent} citation: [{psgs_id}]"
                            )
                            unnecessary_citations.append(psgs_id)
                        else:
                            entail_prec += 1
                    else:
                        entail_prec += 1
            else:
                entail_prec += joint_entail

            eval_log.append(
                {
                    "sent": sent,
                    "target_sent": target_sent,
                    "ref": ref,
                    "joint_entail": joint_entail,
                    "unnecessary_citations": unnecessary_citations,
                }
            )

        sent_total += len(sents)
        ais_scores.append(entail / len(sents))
        ais_scores_prec.append(
            entail_prec / total_citations if total_citations > 0 else 0
        )  # len(sents))

    if sent_mcite > 0 and sent_mcite_support > 0:
        print(
            "Among all sentences, %.2f%% have multiple citations, among which %.2f%% are supported by the joint set, among which %.2f%% overcite."
            % (
                100 * sent_mcite / sent_total,
                100 * sent_mcite_support / sent_mcite,
                100 * sent_mcite_overcite / sent_mcite_support,
            )
        )

    citation_rec = 100 * np.mean(ais_scores)
    citation_prec = 100 * np.mean(ais_scores_prec)

    return {
        "evaluation_logs": eval_log,
        "citation_rec": citation_rec,
        "citation_prec": citation_prec,
    }


def expand_citaions(output):
    """
    Expand citations by following rule:
        1. convert "<sentence 1>. <sentence 2> [2][3]" into "<sentence 1> [2][3]"
        2. "<sentence 1>[1]. <last paragraph senetence>" will be changed to "<sentence 1>[1]. <last paragraph senetence>[1]. "
    """

    def find_citations(sentence):
        return re.findall(r"\[(\d+)\]", sentence)

    modified_pargraphs = []
    for _, paragraph in enumerate(output.split("\n")):
        if len(paragraph) == 0:
            continue
        sentence_endings = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s"
        sentences = re.split(sentence_endings, paragraph)
        sentences = [sentence.strip() for sentence in sentences if sentence]
        modified_sentences = []
        for sentence_idx, sentence in enumerate(sentences):
            if len(sentence) == 0:
                continue
            citations = find_citations(sentence)
            added_citations = ""
            if len(citations) == 0:
                if sentence_idx == len(sentences) - 1 and sentence_idx - 1 >= 0:
                    for citation in find_citations(sentences[sentence_idx - 1]):
                        added_citations += f"[{citation}]"
                elif sentence_idx + 1 < len(sentences):
                    for citation in find_citations(sentences[sentence_idx + 1]):
                        added_citations += f"[{citation}]"
            modified_sentences.append(sentence[:-1] + added_citations + sentence[-1])
        modified_pargraph = " ".join(modified_sentences)
        modified_pargraphs.append(modified_pargraph)
    modified_output = "\n".join(modified_pargraphs).strip()
    return modified_output


def format_data(root_dir, file_name_suffix, do_citation_expansion=False):
    path_polished_article = os.path.join(root_dir, file_name_suffix)
    # print(path_polished_article)
    final_page = load_str(path_polished_article)

    # .../baseline/refined_articles/models--snippet_ranking_model/Boltzmann_Distribution
    url_to_info_path = os.path.join(root_dir, "url_to_info.json")
    search_results = load_json(url_to_info_path)

    url_to_info = {
        value["url"]: {"title": value["title"], "snippets": value["snippets"]}
        for value in search_results["url_to_info"].values()
    }

    output = []
    for line in final_page.split("\n"):
        if len(line) == 0 or line[0] == "#":
            continue
        output.append(line)

    output = "\n".join(output).strip()

    if do_citation_expansion:
        output = expand_citaions(output)

    docs = []

    for url in url_to_info:
        docs.append(
            {
                "title": url_to_info[url]["title"],
                "text": "\n".join(set(url_to_info[url]["snippets"])),
            }
        )

    return output, docs


def main(args):
    global mistral_7b_instruct_vllm

    mistral_7b_instruct_vllm = LLM(
        model=args.model,
        dtype="float16",
        tensor_parallel_size=4,
    )

    if args.mode == "single":
        output, docs = format_data(
            args.dir, args.file_name_suffix, args.do_citation_expansion
        )
        data = [{"output": output, "docs": docs}]
        result = compute_autoais(data=data)

        print("===== Citation Quality =====")
        print(f'recall: {result["citation_rec"]}, precision: {result["citation_prec"]}')

    elif args.mode == "batch":
        df = pd.read_csv(args.batch_topic_path)
        results = {"topic": [], "recall": [], "precision": [], "eval_log": []}
        for i, row in tqdm(df.iterrows(), total=len(df), desc="Batch evaluation"):
            file_name = row["topic"].replace(" ", "_").replace("/", "_")
            output, docs = format_data(
                f"{args.dir}/{file_name}",
                args.file_name_suffix,
                args.do_citation_expansion,
            )
            data = [{"output": output, "docs": docs}]
            result = compute_autoais(data=data)

            results["topic"].append(row["topic"])
            results["recall"].append(result["citation_rec"])
            results["precision"].append(result["citation_prec"])
            results["eval_log"].append(result["evaluation_logs"])

        # Save overall citation quality
        citation_quality_path = os.path.join(
            args.result_output_dir, "citation_quality.json"
        )
        dump_json(results, citation_quality_path)

        # Process and save citation quality grouped by category
        citation_quality_per_category_path = os.path.join(
            args.result_output_dir, "citation_quality_per_category.json"
        )
        citation_quality_per_category = process_citation_quality(results)
        dump_json(citation_quality_per_category, citation_quality_per_category_path)

        # Averaged Aggregated data
        avg_results_citation_quality = {
            "avg_recall": sum(results["recall"]) / len(results["recall"]),
            "avg_precision": sum(results["precision"]) / len(results["precision"]),
        }
        # Define output path and save it
        avg_results_citation_quality_path = os.path.join(
            args.result_output_dir, "avg_citation_quality.json"
        )
        dump_json(avg_results_citation_quality, avg_results_citation_quality_path)

        print(
            f"===== Citation Quality =====\n"
            f"Average recall: {avg_results_citation_quality['avg_recall']}\n"
            f"Average precision: {avg_results_citation_quality['avg_precision']}\n"
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "batch"],
        help="Whether to calculate the precision recall/acc on a single doc or a batch of docs.",
    )
    parser.add_argument(
        "--disable_log", action="store_true", help="Whether to disable log on consoles."
    )
    parser.add_argument("--dir", type=str, help="Directory of the saved results.")
    parser.add_argument(
        "--file_name_suffix", default="", type=str, help="Suffix of the file name."
    )
    parser.add_argument(
        "--batch_topic_path", type=str, help="Path of the file storing batch topics."
    )
    parser.add_argument(
        "--do_citation_expansion",
        action="store_true",
        help="whether expand citations",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "mistralai/Mistral-7B-Instruct-v0.1",
            "mistralai/Mistral-7B-Instruct-v0.2",
            "mistralai/Mistral-7B-Instruct-v0.3",
        ],
        default="mistralai/Mistral-7B-Instruct-v0.1",
        help="Name of the LLM model as a judge. Default is 'mistralai/Mistral-7B-Instruct-v0.1'.",
    )
    parser.add_argument(
        "--fp16", action="store_true", help="Enable FP16/half-precision mode"
    )
    parser.add_argument(
        "--result_output_dir",
        help="Directory to store the evaluation results. "
        "Each article evaluation will be saved as separate file named after {topic_name}.json",
    )
    parser.add_argument("--jobid", type=str, required=False, help="Slurm job ID")

    args = parser.parse_args()
    print(format_args(args))

    if args.disable_log:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    args.result_output_dir = os.path.join(
        args.result_output_dir, args.model.rsplit("/", 1)[-1], args.jobid or ""
    )
    os.makedirs(args.result_output_dir, exist_ok=True)

    """
    python citation_quality.py \
        --mode "batch" \
        --dir "/home/toapantabarahonad/storm-plus/data/baseline/refined_articles/models--snippet_ranking_model" \
        --file_name_suffix "storm_gen_article_polished.txt" \
        --batch_topic_path "../TopicPagesWiki/topics_ores_scores.csv" \
        --do_citation_expansion \
        --result_output_dir "/home/toapantabarahonad/storm-plus/results/storm_citation_eval_results"
    """
    main(args)
