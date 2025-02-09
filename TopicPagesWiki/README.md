# Dataset Construction and Usage

The `TopicPagesWiki` Dataset is a curated collection of high-quality Wikipedia articles aligned with categories from [ScienceDirect Topics](https://www.sciencedirect.com/topics). 

The dataset contains **100 Wikipedia articles**, each corresponding to a ScienceDirect topic. It is designed to assist in writing Wikipedia-like articles from scratch using large language models.

## Scripts and Their Usage

1. **`get_wikipedia_urls.py`**: Retrieves Wikipedia URLs for a list of ScienceDirect Topics.
2. **`get_topics_ores_scores.py`**: Evaluates and filters articles based on quality using the ORES API.
3. **`get_wikipedia_articles.py`**: Parses Wikipedia articles and saves them in multiple formats for analysis.


## How the Dataset Was Built?

For reproducibility, the dataset was constructed using the following steps:

### 1. Retrieving Wikipedia URLs
- **Input**: A list of ScienceDirect Topics.
- **Script**: `get_wikipedia_urls.py`.
- **Output**: A JSON file mapping ScienceDirect topics to their corresponding Wikipedia URLs.

### 2. Filtering High-Quality Articles
- **Purpose**: To include only high-quality Wikipedia articles.
- **Script**: `get_topics_ores_scores.py`.
- **Process**:
  - The script evaluates article quality using the **ORES API**.
  - Articles are classified into the following quality categories: **Stub**, **Start**, **C**, **B**, **GA** (Good Article), and **FA** (Featured Article).
  - Only articles with predicted classes **B**, **GA**, or **FA** are retained.
- **Output**: A CSV and JSON file containing high-quality topics and their corresponding ORES scores.

### 3. Fetching Wikipedia Articles
- **Script**: `get_wikipedia_articles.py`.
- **Process**:
  - Parses the Wikipedia pages for high-quality topics.
  - Extracts the following content:
    - **Structured JSON**: Organizes content into sections and subsections.
    - **Plain Text**: Clean text content.
    - **Markdown**: Formatted text for easy readability.
    - **HTML**: A copy of the raw HTML content.
  - Preserves references and named entities for further analysis.
- **Output**: Multiple formats (`JSON`, `TXT`, `Markdown`, and `HTML`) for each Wikipedia article.

## **Attribution**

This dataset was inspired by the **FreshWiki Dataset**, which focuses on the most-edited Wikipedia pages between February 2022 and September 2023. We acknowledge the authors' contribution and the availability of their data construction pipeline, which formed the basis for some of our processing scripts.

The original dataset and pipeline can be found in the paper:

> Yijia Shao et al. (2024). "Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models." [NAACL HLT 2024](https://arxiv.org/abs/2402.14207).




## Important Links

- [ScienceDirect Topics](https://www.sciencedirect.com/topics)
- [ORES API](https://www.mediawiki.org/wiki/ORES)
- [Wikipedia API](https://www.mediawiki.org/wiki/API:Main_page)