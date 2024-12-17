# from knowledge_storm.rm import DuckDuckGoSearchRM

# def test_ddg_search():

#     rm = DuckDuckGoSearchRM(k=3)  # Get top 3 results
#     results = rm.forward("Fabian Society history")

#     print("Search Results:")
#     for result in results:
#         print(f"\nTitle: {result['title']}")
#         print(f"URL: {result['url']}")
#         print(f"Description: {result['description']}")
#         print(f"Snippets: {result['snippets'][0][:200]}...")

# if __name__ == "__main__":
#     test_ddg_search()

# import logging
# from duckduckgo_search import DDGS

# def test_duckduckgo_search():
#     try:
#         ddgs = DDGS(verify=False)
#         query = "Where is Ecuador located"
#         max_results = 5

#         results = ddgs.text(query, max_results=max_results)

#         print(f"Search results for '{query}':")
#         for i, result in enumerate(results, start=1):
#             print(f"\nResult {i}:")
#             print(f"Title: {result.get('title')}")
#             print(f"URL: {result.get('href')}")
#             print(f"Snippet: {result.get('body')}")

#     except Exception as e:
#         logging.error(f"An error occurred: {e}")

# if __name__ == "__main__":
#     test_duckduckgo_search()

# import logging
# from knowledge_storm.rm import DuckDuckGoSearchRM

# # Custom exception handler for DuckDuckGoSearchRM
# def handle_duckduckgo_error(e):
#     error_message = str(e)  # Extract the error message safely
#     logging.error(f"An error occurred during DuckDuckGo search: {error_message}")

# def test_duckduckgo_search():
#     try:
#         # Initialize DuckDuckGoSearchRM
#         rm = DuckDuckGoSearchRM(k=5)  # Set the number of search results to return

#         query = "Where is Ecuador located"

#         # Use the forward method to perform the search
#         results = rm.forward(query_or_queries=query, exclude_urls=[])

#         print(f"Search results for '{query}':")
#         for i, result in enumerate(results, start=1):
#             print(f"\nResult {i}:")
#             print(f"Title: {result.get('title', 'N/A')}")
#             print(f"URL: {result.get('url', 'N/A')}")
#             print(f"Snippet: {result.get('snippets', ['N/A'])[0]}")  # Get the first snippet if available

#     except Exception as e:
#         handle_duckduckgo_error(e)

# if __name__ == "__main__":
#     test_duckduckgo_search()

import logging
from knowledge_storm.rm import DuckDuckGoSearchRM

logging.basicConfig(level=logging.DEBUG)


def test_duckduckgo_search():
    rm = DuckDuckGoSearchRM(k=5)

    try:
        query = "Where is Ecuador located"
        results = rm.forward(query_or_queries=query)

        print(f"Search results for '{query}':")
        for i, result in enumerate(results, start=1):
            print(f"\nResult {i}:")
            print(f"Title: {result.get('title', 'N/A')}")
            print(f"URL: {result.get('url', 'N/A')}")
            print(f"Snippet: {result.get('snippets', ['N/A'])[0]}")
    except Exception as e:
        logging.error(f"Test failed: {e}")


if __name__ == "__main__":
    test_duckduckgo_search()
