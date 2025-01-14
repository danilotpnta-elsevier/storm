import os
import logging
import wikipediaapi
from config.constants import TOPICS_JSON, DATA_DIR
from src.utils import load_json, dump_json

# Initialize Wikipedia API
username = 'Knowledge Curation Bot'
wiki_wiki = wikipediaapi.Wikipedia(username, "en")

# Set up logging
logging.basicConfig(
    filename="wiki_url_retrieval.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
)


def get_wikipedia_url(topic):
    """
    Retrieve the Wikipedia URL for a given topic.
    Handles cases where the title does not directly match.
    """
    # clean topic in case of special characters
    topic = topic.capitalize() #TODO: chec if other cleaning methods are needed

    try:
        # Directly fetch the page
        page = wiki_wiki.page(topic)

        # If the page exists, return the URL
        if page.exists():
            logging.info(
                f"Found exact match for topic '{topic}' at URL: {page.fullurl}"
            )
            return page.fullurl

        # Handle abbreviation or alternative titles
        search_results = wiki_wiki.search(topic, results=5)
        if search_results:
            print(search_results)
            for result in search_results:
                result_page = wiki_wiki.page(result)
                if result_page.exists():
                    logging.info(
                        f"Found alternative match for topic '{topic}' as '{result}' at URL: {result_page.fullurl}"
                    )
                    return result_page.fullurl

        # If no match is found
        logging.warning(f"No Wikipedia page found for topic: '{topic}'")
        return None

    except Exception as e:
        logging.error(f"Error retrieving Wikipedia URL for topic '{topic}': {e}")
        return None


def mk_topic_urls_dict(topics):
    """
    Create a dictionary of topics and their corresponding Wikipedia URLs.
    """
    topic_urls = {}
    for topic in topics:
        url = get_wikipedia_url(topic)
        topic_urls[topic] = url
    return topic_urls


def main():

    topics = load_json(TOPICS_JSON)
    topic_urls = mk_topic_urls_dict(topics)

    # Log the results to a file or console
    for topic, url in topic_urls.items():
        if url:
            # print(f"Topic: {topic} - URL: {url}")
            void = 1
        else:
            print(f"Topic: {topic} - No URL found.")

    print(f"Found {len([url for url in topic_urls.values() if url])} URLs out of {len(topic_urls)} topics.")

    save_path = os.path.join(DATA_DIR, "topic_urls.json")
    dump_json(topic_urls, save_path)
    

    # For looking at topics that were not found by this script
    # here you can dicern if the Wikipedia Search finds the topic:
    # https://en.wikipedia.org/w/index.php?search

if __name__ == "__main__":
    main()