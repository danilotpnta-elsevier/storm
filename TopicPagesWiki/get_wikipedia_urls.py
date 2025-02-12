import os
import time
import random
import argparse
from tqdm import tqdm

import wikipediaapi
import wikipedia

from concurrent.futures import ThreadPoolExecutor, as_completed
from config.constants import TOPICS_ALL_JSON, data_dir
from src.utils import load_json, dump_json, format_args


class WikiURLFetcher:
    def __init__(self):
        self.USERNAME = "Knowledge Curation Bot"
        self.wiki_wiki = wikipediaapi.Wikipedia(self.USERNAME, "en")
        self.rate_limit_window = 1.0
        self.requests_per_window = 10
        self.current_window_requests = 0
        self.last_request_time = time.time()

    def _rate_limit(self):
        """
        Prevents hitting Wikipedia's request limits
        """
        current_time = time.time()
        if current_time - self.last_request_time >= self.rate_limit_window:
            self.current_window_requests = 0
            self.last_request_time = current_time

        if self.current_window_requests >= self.requests_per_window:
            sleep_time = self.rate_limit_window - (
                current_time - self.last_request_time
            )
            if sleep_time > 0:
                time.sleep(sleep_time + random.uniform(0.1, 0.5))
            self.current_window_requests = 0
            self.last_request_time = time.time()

        self.current_window_requests += 1

    def get_wikipedia_url(self, topic, retries=3):
        """
        Retrieve the Wikipedia URL for a given topic.
        Handles cases where the title does not directly match.
        """
        topic = topic.capitalize()

        for attempt in range(retries):
            try:
                self._rate_limit()
                print(
                    f"🔍 Fetching: {topic} (Attempt {attempt+1}/{retries})", flush=True
                )

                # Try exact match first
                page = self.wiki_wiki.page(topic)
                if page.exists():
                    print(f"✔ Exact Match: {topic} -> {page.fullurl}")
                    return topic, page.fullurl, True

                # Try search, if no exact match
                # Note: may not always return the best match
                # e.g. ~/notebooks/fetching_wiki_urls.ipynb
                self._rate_limit()
                page_title = wikipedia.page(topic, auto_suggest=True).title
                url = f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}"
                print(f"✔ Fallback Match: {topic} -> {url}")
                return topic, url, False

            except wikipedia.exceptions.DisambiguationError as e:
                print(
                    f"⚠️ Disambiguation: {topic} → Trying '{e.options[0]}'", flush=True
                )
                first_option = e.options[0]
                return self.get_wikipedia_url(first_option, retries=1)

            except wikipedia.exceptions.PageError:
                if attempt == retries - 1:
                    print(f"❌ No page found for: {topic}", flush=True)
                if attempt < retries - 1:
                    backoff = min(2**attempt, 8)
                    time.sleep(backoff + random.uniform(0.1, 0.5))
                continue

            except Exception as e:
                print(
                    f"⚠️ Error fetching '{topic}' (Attempt {attempt+1}/{retries}): {e}",
                    flush=True,
                )
                if attempt < retries - 1:
                    time.sleep(2**attempt + random.uniform(0.1, 0.5))
                continue

        return topic, None, None


def process_topics(topics, fetcher, args):

    exact_matches = {}
    fallback_matches = {}
    save_path_exact = os.path.join(
        args.output_dir, f"{args.output_file_name}_urls_exact_matches.json"
    )
    save_path_fallback = os.path.join(
        args.output_dir, f"{args.output_file_name}_urls_fallback_matches.json"
    )

    # If resuming loads existing data
    if os.path.exists(save_path_exact):
        exact_matches = load_json(save_path_exact)
        print(f"Resuming {len(exact_matches)} exact matches")

    if os.path.exists(save_path_fallback):
        fallback_matches = load_json(save_path_fallback)
        print(f"Resuming {len(fallback_matches)} fallback matches")

    topics_to_process = [
        topic
        for topic in topics
        if topic not in exact_matches and topic not in fallback_matches
    ]

    if args.max_workers is None:
        for topic in tqdm(topics_to_process, desc="Fetching Wikipedia URLs"):
            topic, url, is_exact = fetcher.get_wikipedia_url(topic)
            if is_exact is True:
                exact_matches[topic] = url
            elif is_exact is False:
                fallback_matches[topic] = url

    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(fetcher.get_wikipedia_url, topic): topic
                for topic in topics_to_process
            }
            with tqdm(
                total=len(topics_to_process), desc="Fetching Wikipedia URLs"
            ) as pbar:
                for future in as_completed(futures):
                    topic, url, is_exact = future.result()
                    if is_exact is True:
                        exact_matches[topic] = url
                    elif is_exact is False:
                        fallback_matches[topic] = url
                    pbar.update(1)

                    # Save every 50 entries
                    if (len(exact_matches) + len(fallback_matches)) % 50 == 0:
                        dump_json(exact_matches, save_path_exact)
                        dump_json(fallback_matches, save_path_fallback)

    # Final save
    dump_json(exact_matches, save_path_exact)
    dump_json(fallback_matches, save_path_fallback)

    # Compute statistics
    total_processed = len(exact_matches) + len(fallback_matches)
    success_rate = (
        (len(exact_matches) / total_processed * 100) if total_processed > 0 else 0.0
    )
    print(
        f"\n=============== SUMMARY ===============\n"
        f"Total topics processed: {total_processed}\n"
        f"Exact matches: {len(exact_matches)}\n"
        f"Fallback matches: {len(fallback_matches)}\n"
        f"Success rate: {success_rate:.1f}%\n"
        f"\nExact matches saved to: {save_path_exact}\n"
        f"Fallback matches saved to: {save_path_fallback}\n"
    )


def main(args):
    topics = load_json(args.topics_json)
    fetcher = WikiURLFetcher()

    print(f"Processing {len(topics)} topics with {args.max_workers} workers...")
    process_topics(topics, fetcher, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retrieve Wikipedia URLs for a list of topics."
    )
    parser.add_argument(
        "--topics_json",
        type=str,
        default=TOPICS_ALL_JSON,
        help="Path to the JSON file containing topics.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=data_dir,
        help="Directory to save output JSON.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        required=False,
        help="Number of parallel workers",
    )
    args = parser.parse_args()
    print(format_args(args))

    os.makedirs(args.output_dir, exist_ok=True)
    args.output_file_name = os.path.splitext(os.path.basename(args.topics_json))[0]

    if args.max_workers is not None:
        args.max_workers = min(args.max_workers, os.cpu_count())

    main(args)
