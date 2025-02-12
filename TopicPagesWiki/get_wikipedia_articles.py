import os
import re
import argparse
import pandas as pd
from tqdm import tqdm

import requests
import wikipediaapi
from bs4 import BeautifulSoup

from flair.data import Sentence
from flair.nn import Classifier

from config.constants import TOPICS_ORES_SCORES_JSON, TOPICS_URLS_JSON
from src.utils import (
    load_json,
    write_html,
    dump_json,
    write_str,
    md_to_pdf,
    format_args,
)

from typing import List


def get_references(sentence, reference_dict):
    """
    Given a sentence, extract all reference index and find links from dictionary,
    then remove reference brackets from original sentences

    @param sentence, sentence to process
    @param reference_dict, dictionary of references
    @return cleaned sentence, reference_list pair
    """
    refs = re.findall(r"\[\d+\]", sentence)
    sentence = re.sub(r"\[\d+\]", "", sentence).strip().replace("\n", "")
    return sentence, [
        reference_dict[ref.replace("[", "").replace("]", "")] for ref in refs
    ]


def load_topics():
    """Load topics from json file"""

    # TODO: Investigate whether this would be more optimal
    # topic_ores_scores_dict = load_json(TOPICS_ORES_SCORES_JSON)
    # urls_topics_dict = {entry["url"]: entry["topic"] for entry in topic_ores_scores_dict}

    topic_urls_dict = load_json(TOPICS_URLS_JSON)
    urls_topics_dict = {url: topic for topic, url in topic_urls_dict.items()}
    return urls_topics_dict


def get_content_until_next_header(element):
    """Helper function to get all content until next header"""
    content = []
    current = element.next_sibling

    while current:
        if current.name == "p":
            content.append(current)

        elif current.name == "div":
            headers = current.find_all(
                ["h1", "h2", "h3", "h4", "h5", "h6"], recursive=True
            )
            if headers:
                break

        current = current.next_sibling
    return content


def extract_data(html, reference_dict):
    """
    Extract section data from wiki url.

    @param url: wiki url
    @reference_dict, reference dict from extract_references()
    @return a dictionary, key is section / subsection name, value is a list of {"sentence": ..., "refs": []}
    """

    data = {}

    for header in html.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        header_container = header.parent
        section_title = header.text.replace("[edit]", "").strip().replace("\xa0", " ")
        section_data = []

        paragraphs = get_content_until_next_header(header_container)

        for p in paragraphs:
            for sentence in p.text.replace("[", " [").split(". "):
                if sentence:
                    sentence, refs = get_references(sentence, reference_dict)
                    if sentence:
                        section_data.append({"sentence": sentence, "refs": refs})

        data[section_title] = section_data

    return data


def extract_references(html):
    """
    Extract references from reference section with following structure:
    {
      "1": "https://...",
      "2": "https://...",
    }

    @param html wikipedia page
    @return dictionary of references, key is the citation index string, value is correpsonding url link
    """

    references = {}
    for references_section in html.find_all("ol", {"class": "references"}):
        for ref in references_section.find_all(
            "li", {"id": lambda x: x and x.startswith("cite_note-")}
        ):
            index = str(ref["id"].rsplit("-", 1)[-1])
            link_tag = ref.find("a", {"href": True, "rel": "nofollow"})
            isbn_tag = ref.find("a", href=re.compile(r"Special:BookSources"))
            if link_tag:
                link = link_tag["href"]
                references[index] = link
            elif isbn_tag:
                link = f"ISBN: {isbn_tag.text}"
                references[index] = link
            else:
                # TODO: handle other types of references
                # Danilo: For the time being this is not necessary for computing metrics
                references[index] = "[ERROR retrieving ref link]"
    return references


def getSections(page, structured_data):
    """
    Recursively extract each section title and plain text.

    @param page, page variable from wikipediaapi (e.g. wiki_api.page("page name"))
    @return a list of nested json for each section and corresponding subsections
    {
        "section_title": ...,
        "section_text": ...,
        "subsections": [
            {...},
            {...}
        ]
    }
    """
    return [
        {
            "section_title": i.title,
            "section_content": structured_data[i.title],
            "subsections": getSections(i, structured_data),
        }
        for i in page.sections
    ]


def fetch_data_wikipedia(username, url):
    """
    Get wikepdia output as format json

    @param username, username for wikipedia api agent.
    @param url, url of wikipedia page
    """
    wiki_api = wikipediaapi.Wikipedia(username, "en")
    wikipedia_page_name = url.replace("https://en.wikipedia.org/wiki/", "")
    wikiapi_page = wiki_api.page(wikipedia_page_name)

    response = requests.get(url)
    html_page = BeautifulSoup(response.content, "html.parser")

    # extract references
    reference_dict = extract_references(html_page)
    structured_data = extract_data(html_page, reference_dict)

    # save extracted result to file
    extracted_data = {
        "title": wikipedia_page_name,
        "url": url,
        "summary": wikiapi_page.summary,
        "content": getSections(wikiapi_page, structured_data),
        "references": reference_dict,
    }

    return html_page, extracted_data, reference_dict


def section_dict_to_text(data, inv_reference_dict, level=1):
    title = data["section_title"]
    content = data["section_content"]
    subsections = data["subsections"]
    if len(content) == 0 and len(subsections) == 0:
        return ""
    result = f"\n\n{'#' * level} {title}"
    if content:
        result += "\n\n"
        for cur_sentence in content:
            result += cur_sentence["sentence"]
            if cur_sentence["refs"]:
                result += " "
                result += " ".join(
                    f"[{inv_reference_dict[ref]}]"
                    for ref in cur_sentence["refs"]
                    if ref != "[ERROR retrieving ref link]"
                )
            result += ". "
    for subsection in subsections:
        result += section_dict_to_text(subsection, inv_reference_dict, level=level + 1)
    return result


def output_as_text(result, reference_dict):
    inv_reference_dict = {v: k for k, v in reference_dict.items()}
    output = result["title"] + "\n\n"
    output += result["summary"]
    for section in result["content"]:
        output += section_dict_to_text(section, inv_reference_dict)
    output += "\n\n# References\n\n"
    for idx, link in reference_dict.items():
        output += f"[{idx}] {link}\n"
    return output


tagger = Classifier.load("ner")


def extract_entities_flair(text):
    clean_txt = re.sub(
        r"#+ ", "", re.sub(r"\[\d+\]", "", text[: text.find("\n\n# References\n\n")])
    )
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", clean_txt)
    entities = []
    for sentence in sentences:
        if len(sentence) == 0:
            continue
        sentence = Sentence(sentence)
        tagger.predict(sentence)
        entities.extend([entity.text for entity in sentence.get_spans("ner")])

    entities = list(set([e.lower() for e in entities]))

    return entities


class FilesHandler:
    def __init__(self, output_dir: str, files_types_to_save: List[str]):
        self.output_dir = output_dir
        self.files_types_to_save = files_types_to_save
        self.init_output_dir = self.init_output_dir()

    def init_output_dir(self):
        for file_type in self.files_types_to_save:
            os.makedirs(os.path.join(self.output_dir, file_type), exist_ok=True)
            print(f"Saving files to: ", os.path.join(self.output_dir, file_type))

    def save(self, topic, html_page, txt, result):

        topic = topic.replace(" ", "_")

        # 1. Saves defaults: html, txt, json
        write_html(
            str(html_page.prettify()),
            os.path.join(self.output_dir, "html", topic + ".html"),
        )
        write_str(txt, os.path.join(self.output_dir, "txt", topic + ".txt"))
        dump_json(result, os.path.join(self.output_dir, "json", topic + ".json"))

        # 2. Saves extra files
        if "md" in self.files_types_to_save:
            md_path = os.path.join(self.output_dir, "md", topic + ".md")
            write_str(txt, md_path)

            # TODO: check if this ouputs error
            if "pdf" in self.files_types_to_save:
                md_to_pdf(md_path, os.path.join(self.output_dir, "pdf", topic + ".pdf"))


def process_url(url: str, username: str = "Knowledge Curation Project"):
    """
    Process a Wikipedia page and save the result to the output directory.
    """

    html_page, result, reference_dict = fetch_data_wikipedia(username=username, url=url)
    txt = output_as_text(result, reference_dict)
    result["flair_entities"] = extract_entities_flair(txt)

    return html_page, txt, result


def main(args):

    urls_topics_dict = load_topics()
    fileManager = FilesHandler(args.output_dir, args.files_types_to_save)

    if args.batch_path:
        df = pd.read_csv(args.batch_path)
        for _, row in tqdm(
            df.iterrows(), total=len(df), desc="Fetching Wikipedia pages"
        ):
            try:
                url = row["url"]
                topic = urls_topics_dict.get(url, url.split("/")[-1])
                html_page, txt, result = process_url(url)
                fileManager.save(topic, html_page, txt, result)

            except Exception as e:
                print(e)
                print(f'Error occurs when processing {row["url"]}')
    else:
        topic = urls_topics_dict.get(args.url, args.url.split("/")[-1])
        html_page, txt, result = process_url(args.url)
        fileManager.save(topic, html_page, txt, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse a Wikipedia page into sections."
    )
    parser.add_argument(
        "--batch_path",
        required=False,
        type=str,
        help="Path to CSV file containing URLs to parse.",
    )
    parser.add_argument(
        "-u",
        "--url",
        # default="https://en.wikipedia.org/wiki/Python_(programming_language)",
        # default="https://en.wikipedia.org/wiki/Wave",
        default="https://en.wikipedia.org/wiki/Zillennials",
        help="The URL of the Wikipedia page to parse (default: https://en.wikipedia.org/wiki/Python_(programming_language))",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default="./",
        help="The directory where the parsed content will be saved (default: current directory)",
    )
    parser.add_argument(
        "--files_types_to_save",
        nargs="+",
        choices=["html", "txt", "json", "md", "pdf"],
        help="The types of files to save (Default: html, txt and json). Additional formats will be saved.",
    )

    args = parser.parse_args()
    defaults = ["html", "txt", "json"]
    args.files_types_to_save = list(set(defaults + args.files_types_to_save))
    print(args.files_types_to_save)
    print(format_args(args))

    """
    python wikipage_extractor.py --batch_path "/home/toapantabarahonad/ds-agentic-topic-pages-gen/storm/TopicPagesWiki/topics_ores_scores.csv"
    """

    main(args)
