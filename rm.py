import os
from typing import Union, List

import dspy

from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient, models
from langchain_huggingface import HuggingFaceEmbeddings


class VectorRM(dspy.Retrieve):
    """Retrieve information from custom documents using Qdrant.

    To be compatible with STORM, the custom documents should have the following fields:
        - content: The main text content of the document.
        - title: The title of the document.
        - url: The URL of the document. STORM use url as the unique identifier of the document, so ensure different
            documents have different urls.
        - description (optional): The description of the document.
    The documents should be stored in a CSV file.
    """

    def __init__(
        self,
        collection_name: str,
        embedding_model: str,
        device: str = "mps",
        k: int = 3,
        seed: int = None,
        query_prefix: str = "query:",
    ):
        """
        Params:
            collection_name: Name of the Qdrant collection.
            embedding_model: Name of the Hugging Face embedding model.
            device: Device to run the embeddings model on, can be "mps", "cuda", "cpu".
            k: Number of top chunks to retrieve.
            seed: Seed for deterministic behavior.
            query_prefix: Prefix to add to the query before embedding.
        """
        super().__init__(k=k)
        self.seed = seed
        self.usage = 0
        self.query_prefix = query_prefix
        self.embedding_model = embedding_model
        self.filter_condition = None

        if not collection_name:
            raise ValueError("Please provide a collection name.")
        if not embedding_model:
            raise ValueError("Please provide an embedding model.")

        model_kwargs = {"device": device, "trust_remote_code": True}
        encode_kwargs = {"normalize_embeddings": True}
        self.model = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

        self.collection_name = collection_name
        self.client = None
        self.qdrant = None
        self.cache = {}

        if self.seed is not None:
            print(f"Initializing deterministic VectorRM with seed {self.seed}")
            self._make_deterministic()

    def set_filter_by(self, title: str):
        """
        Creates a filter condition for searching documents by title.

        Args:
            title (str): The title to filter by when retrieving documents.
        """

        if title:
            self.filter_condition = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.title",
                        match=models.MatchValue(value=title),
                    )
                ]
            )

    def _check_collection(self):
        """
        Check if the Qdrant collection exists and create it if it does not.
        """
        if self.client is None:
            raise ValueError("Qdrant client is not initialized.")
        if self.client.collection_exists(collection_name=f"{self.collection_name}"):
            print(
                f"Collection {self.collection_name} exists. Loading the collection..."
            )
            self.qdrant = Qdrant(
                client=self.client,
                collection_name=self.collection_name,
                embeddings=self.model,
            )
        else:
            raise ValueError(
                f"Collection {self.collection_name} does not exist. Please create the collection first."
            )

    def init_docker_qdrant(self):
        """Initialize the Qdrant client that is connected to a Docker instance."""
        try:
            self.client = QdrantClient("localhost", port=6333)
            self.search_params = models.SearchParams(
                quantization=models.QuantizationSearchParams(
                    ignore=False,
                    rescore=True,
                    oversampling=1.0,
                )
            )
            self._check_collection()
        except Exception as e:
            raise ValueError(f"Error connecting to Docker Qdrant: {e}")

    def init_offline_vector_db(self, vector_store_path: str):
        """
        Initialize the Qdrant client that is connected to an offline vector store with the given vector store folder path.

        Args:
            vector_store_path (str): Path to the vector store.
        """
        if vector_store_path is None:
            raise ValueError("Please provide a folder path.")

        try:
            self.client = QdrantClient(path=vector_store_path)
            self.search_params = None
            self._check_collection()
        except Exception as e:
            raise ValueError(f"Error occurs when loading the vector store: {e}")

    def get_usage_and_reset(self):
        usage = self.usage
        self.usage = 0

        return {"VectorRM": usage}

    def cleanup(self):
        """Release resources when done with this retriever."""
        self.client = None
        self.qdrant = None
        self.filter_condition = None

    def get_vector_count(self):
        """
        Get the count of vectors in the collection.

        Returns:
            int: Number of vectors in the collection.
        """
        return self.qdrant.client.count(collection_name=self.collection_name)

    def _make_deterministic(self):
        """Configure the retriever for deterministic behavior."""
        import types

        if not hasattr(self, "_original_forward"):
            self._original_forward = self.forward

        def deterministic_forward(self, query_or_queries, exclude_urls=None):
            """Deterministic version of the retrieval method."""
            queries = (
                [query_or_queries]
                if isinstance(query_or_queries, str)
                else query_or_queries
            )
            queries = sorted(queries)
            self.usage += len(queries)

            collected_results = []
            for query in queries:
                if (
                    "snowflake" in self.embedding_model.lower()
                    or "arctic" in self.embedding_model.lower()
                ):
                    if not query.strip():
                        print("Empty query received!")
                    query = self.query_prefix + query

                if self.qdrant is None:
                    print("Warning: Qdrant is not initialized")
                    continue

                if self.filter_condition:
                    related_docs = self.qdrant.similarity_search_with_score(
                        query=query,
                        k=self.k,
                        filter=self.filter_condition,
                        search_params=self.search_params,
                    )
                else:
                    related_docs = self.qdrant.similarity_search_with_score(
                        query,
                        k=self.k,
                        search_params=self.search_params,
                    )

                related_docs = sorted(
                    related_docs, key=lambda x: (-x[1], x[0].metadata.get("url", ""))
                )

                for doc, _ in related_docs:
                    result = {
                        "description": doc.metadata.get("description", ""),
                        "snippets": [doc.page_content],
                        "title": doc.metadata.get("title", ""),
                        "url": doc.metadata.get("url", ""),
                    }
                    collected_results.append(result)
            return collected_results

        self.forward = types.MethodType(deterministic_forward, self)

    def forward(
        self,
        query_or_queries: Union[str, List[str]],
        exclude_urls: List[str],
    ):
        """
        Search in your data for self.k top passages for query or queries.

        Args:
            query_or_queries (Union[str, List[str]]): The query or queries to search for.
            exclude_urls (List[str]): Dummy parameter to match the interface. Does not have any effect.

        Returns:
            a list of Dicts, each dict has keys of 'description', 'snippets' (list of strings), 'title', 'url'
        """
        queries = (
            [query_or_queries]
            if isinstance(query_or_queries, str)
            else query_or_queries
        )
        self.usage += len(queries)
        collected_results = []

        for query in queries:
            if (
                "snowflake" in self.embedding_model.lower()
                or "arctic" in self.embedding_model.lower()
            ):
                if not query.strip():
                    print("Empty query received!")
                query = self.query_prefix + query

            if self.filter_condition:
                related_docs = self.qdrant.similarity_search_with_score(
                    query=query,
                    k=self.k,
                    filter=self.filter_condition,
                    search_params=self.search_params,
                )
            else:
                related_docs = self.qdrant.similarity_search_with_score(
                    query,
                    k=self.k,
                    search_params=self.search_params,
                )

            for i in range(len(related_docs)):
                doc = related_docs[i][0]
                collected_results.append(
                    {
                        "description": doc.metadata["description"],
                        "snippets": [doc.page_content],
                        "title": doc.metadata["title"],
                        "url": doc.metadata["url"],
                    }
                )

        return collected_results
