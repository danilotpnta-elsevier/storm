import concurrent.futures
import logging
import os
from concurrent.futures import as_completed
from typing import Union, List, Tuple, Optional, Dict

import dspy

from .callback import BaseCallbackHandler
from .persona_generator import StormPersonaGenerator
from .storm_dataclass import DialogueTurn, StormInformationTable
from ...interface import KnowledgeCurationModule, Retriever, Information
from ...utils import ArticleTextProcessing

try:
    from streamlit.runtime.scriptrunner import add_script_run_ctx

    streamlit_connection = True
except ImportError as err:
    streamlit_connection = False

script_dir = os.path.dirname(os.path.abspath(__file__))


class ConvSimulator(dspy.Module):
    """Simulate a conversation between a Wikipedia writer with specific persona and an expert."""

    def __init__(
        self,
        topic_expert_engine: Union[dspy.LM],
        question_asker_engine: Union[dspy.LM],
        retriever: Retriever,
        max_search_queries_per_turn: int,
        search_top_k: int,
        max_turn: int,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.wiki_writer = WikiWriter(engine=question_asker_engine, seed=seed)
        self.topic_expert = TopicExpert(
            engine=topic_expert_engine,
            max_search_queries=max_search_queries_per_turn,
            search_top_k=search_top_k,
            retriever=retriever,
            seed=seed,
        )
        self.max_turn = max_turn
        self.seed = seed

    def forward(
        self,
        topic: str,
        persona: str,
        ground_truth_url: str,
        callback_handler: BaseCallbackHandler,
    ):
        """
        topic: The topic to research.
        persona: The persona of the Wikipedia writer.
        ground_truth_url: The ground_truth_url will be excluded from search to avoid ground truth leakage in evaluation.
        """
        dlg_history: List[DialogueTurn] = []
        for _ in range(self.max_turn):
            user_utterance = self.wiki_writer(
                topic=topic, persona=persona, dialogue_turns=dlg_history
            ).question
            if user_utterance == "":
                logging.error("Simulated Wikipedia writer utterance is empty.")
                break
            if user_utterance.startswith("Thank you so much for your help!"):
                break
            print("- User utterance: ", user_utterance)
            expert_output = self.topic_expert(
                topic=topic, question=user_utterance, ground_truth_url=ground_truth_url
            )
            dlg_turn = DialogueTurn(
                agent_utterance=expert_output.answer,
                user_utterance=user_utterance,
                search_queries=expert_output.queries,
                search_results=expert_output.searched_results,
            )
            dlg_history.append(dlg_turn)
            callback_handler.on_dialogue_turn_end(dlg_turn=dlg_turn)

        return dspy.Prediction(dlg_history=dlg_history)


class WikiWriter(dspy.Module):
    """Perspective-guided question asking in conversational setup.

    The asked question will be used to start a next round of information seeking."""

    def __init__(self, engine: Union[dspy.LM], seed: Optional[int] = None):
        super().__init__()
        self.ask_question_with_persona = dspy.ChainOfThought(AskQuestionWithPersona)
        self.ask_question = dspy.ChainOfThought(AskQuestion)
        self.engine = engine
        self.seed = seed

    def forward(
        self,
        topic: str,
        persona: str,
        dialogue_turns: List[DialogueTurn],
        draft_page=None,
    ):
        conv = []
        for turn in dialogue_turns[:-4]:
            conv.append(
                f"You: {turn.user_utterance}\nExpert: Omit the answer here due to space limit."
            )
        for turn in dialogue_turns[-4:]:
            conv.append(
                f"You: {turn.user_utterance}\nExpert: {ArticleTextProcessing.remove_citations(turn.agent_utterance)}"
            )
        conv = "\n".join(conv)
        conv = conv.strip() or "N/A"
        conv = ArticleTextProcessing.limit_word_count_preserve_newline(conv, 2500)

        with dspy.settings.context(lm=self.engine, seed=self.seed):
            if persona is not None and len(persona.strip()) > 0:
                question = self.ask_question_with_persona(
                    topic=topic, persona=persona, conv=conv
                ).question
            else:
                question = self.ask_question(
                    topic=topic, persona=persona, conv=conv
                ).question

        return dspy.Prediction(question=question)


class AskQuestion(dspy.Signature):
    """You are an experienced Wikipedia writer. You are chatting with an expert to get information for the topic you want to contribute. Ask good questions to get more useful information relevant to the topic.
    When you have no more question to ask, say "Thank you so much for your help!" to end the conversation.
    Please only ask a question at a time and don't ask what you have asked before. Your questions should be related to the topic you want to write.
    """

    topic = dspy.InputField(prefix="Topic you want to write: ", format=str)
    conv = dspy.InputField(prefix="Conversation history:\n", format=str)
    question = dspy.OutputField(format=str)


class AskQuestionWithPersona(dspy.Signature):
    """You are an experienced Wikipedia writer and want to edit a specific page. Besides your identity as a Wikipedia writer, you have specific focus when researching the topic.
    Now, you are chatting with an expert to get information. Ask good questions to get more useful information.
    When you have no more question to ask, say "Thank you so much for your help!" to end the conversation.
    Please only ask a question at a time and don't ask what you have asked before. Your questions should be related to the topic you want to write.
    """

    topic = dspy.InputField(prefix="Topic you want to write: ", format=str)
    persona = dspy.InputField(
        prefix="Your persona besides being a Wikipedia writer: ", format=str
    )
    conv = dspy.InputField(prefix="Conversation history:\n", format=str)
    question = dspy.OutputField(format=str)


class QuestionToQuery(dspy.Signature):
    """You want to answer the question using Google search. What do you type in the search box?
    Write the queries you will use in the following format:
    - query 1
    - query 2
    ...
    - query n"""

    topic = dspy.InputField(prefix="Topic you are discussing about: ", format=str)
    question = dspy.InputField(prefix="Question you want to answer: ", format=str)
    queries = dspy.OutputField(format=str)


class AnswerQuestion(dspy.Signature):
    """You are an expert who can use information effectively. You are chatting with a Wikipedia writer who wants to write a Wikipedia page on topic you know. You have gathered the related information and will now use the information to form a response.
    Make your response as informative as possible, ensuring that every sentence is supported by the gathered information. If the [gathered information] is not directly related to the [topic] or [question], provide the most relevant answer based on the available information. If no appropriate answer can be formulated, respond with, “I cannot answer this question based on the available information,” and explain any limitations or gaps.
    """

    topic = dspy.InputField(prefix="Topic you are discussing about:", format=str)
    conv = dspy.InputField(prefix="Question:\n", format=str)
    info = dspy.InputField(prefix="Gathered information:\n", format=str)
    answer = dspy.OutputField(
        prefix="Now give your response. (Try to use as many different sources as possible and add do not hallucinate.)\n",
        format=str,
    )


class TopicExpert(dspy.Module):
    """Answer questions using search-based retrieval and answer generation. This module conducts the following steps:
    1. Generate queries from the question.
    2. Search for information using the queries.
    3. Filter out unreliable sources.
    4. Generate an answer using the retrieved information.
    """

    def __init__(
        self,
        engine: Union[dspy.LM],
        max_search_queries: int,
        search_top_k: int,
        retriever: Retriever,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.generate_queries = dspy.Predict(QuestionToQuery)
        self.retriever = retriever
        self.answer_question = dspy.Predict(AnswerQuestion)
        self.engine = engine
        self.max_search_queries = max_search_queries
        self.search_top_k = search_top_k
        self.seed = seed

    def forward(self, topic: str, question: str, ground_truth_url: str):
        # 1. If seed is provided, use it in dspy context.
        context_kwargs = {"lm": self.engine, "show_guidelines": False}
        if self.seed is not None:
            context_kwargs["seed"] = self.seed
            print(f"[TopicExpert.forward] Using seed={self.seed} -> deterministic mode")

        with dspy.settings.context(**context_kwargs):
            # 2. Generate queries
            queries_str = self.generate_queries(topic=topic, question=question).queries
            raw_queries = [
                q.replace("-", "").strip().strip('"').strip()
                for q in queries_str.split("\n")
            ]
            raw_queries = raw_queries[: self.max_search_queries]

            # 3. Sort queries only if we have a seed
            if self.seed is not None:
                raw_queries = sorted(dict.fromkeys(raw_queries))
                print(f"[seed={self.seed}] Queries (sorted): {raw_queries}")
            else:
                # Non-deterministic dedup
                raw_queries = list(set(raw_queries))

            # 4. Perform retrieval
            searched_results = self.retriever.retrieve(
                query=raw_queries,
                exclude_urls=[ground_truth_url],
            )
            logging.info(f"Retrieved {len(searched_results)} results total.")

            # 5. Sort final results if seed is set
            if self.seed is not None:
                # For each result, let's log snippet list before sorting
                for i, info_obj in enumerate(searched_results):
                    logging.info(
                        f"Before sorting, result #{i} URL={info_obj.url}, snippet_count={len(info_obj.snippets)}"
                    )
                    for sn_i, snippet in enumerate(info_obj.snippets):
                        logging.info(f"   snippet[{sn_i}]: {snippet[:80]!r}...")


                # Sort each snippet list
                for info_obj in searched_results:
                    if info_obj.snippets:
                        print(f"Before sorting, URL: {info_obj.url}")
                        print(f"Before sorting, snippets: {info_obj.snippets}")
                        info_obj.snippets = sorted(info_obj.snippets)
                        print(f"After sorting, snippets: {info_obj.snippets}")

                # Then sort the entire list, e.g. by (url, snippet[0], etc.)
                searched_results.sort(
                    key=lambda info: (
                        info.url or "",
                        info.snippets[0] if info.snippets else "",
                    )
                )

                # Log the sorted snippet lists
                for i, info_obj in enumerate(searched_results):
                    logging.info(
                        f"After sorting, result #{i} URL={info_obj.url}, snippet_count={len(info_obj.snippets)}"
                    )
                    for sn_i, snippet in enumerate(info_obj.snippets):
                        logging.info(f"   sorted snippet[{sn_i}]: {snippet[:80]!r}...")

            # 6. Build info text, generate an answer
            if len(searched_results) == 0:
                answer = "Sorry, I cannot find information for this question."
            else:
                info = ""
                # Let's also log the order in which we use the top_k results
                top_k_results = searched_results[: self.search_top_k]
                logging.info(f"Using top {self.search_top_k} results for final answer.")
                for i, r in enumerate(top_k_results):
                    logging.info(f"   (rank {i}) -> URL={r.url}")
                    if r.snippets:
                        snippet_text = r.snippets[0]
                        info += f"[{i+1}]: {snippet_text}\n\n"
                info = ArticleTextProcessing.limit_word_count_preserve_newline(info, 1000)

                try:
                    answer = self.answer_question(topic=topic, conv=question, info=info).answer
                    answer = ArticleTextProcessing.remove_uncompleted_sentences_with_citations(answer)
                except Exception as e:
                    logging.error(f"Error in answer generation: {e}")
                    answer = "Sorry, I cannot answer this question."

        return dspy.Prediction(
            queries=raw_queries,
            searched_results=searched_results,
            answer=answer
        )

    # def forward(self, topic: str, question: str, ground_truth_url: str):
    #     seed = getattr(self, "seed", None)
    #     print("Seed is ", seed)
    #     if hasattr(self.engine, "kwargs"):
    #         print("Seed of self.engine is ", self.engine.kwargs.get("seed"))

    #     # Use seed in context if provided
    #     context_kwargs = {"lm": self.engine, "show_guidelines": False}
    #     if seed is not None:
    #         context_kwargs["seed"] = seed

    #     with dspy.settings.context(**context_kwargs):
    #         # Generate queries
    #         queries = self.generate_queries(topic=topic, question=question).queries
    #         queries = [
    #             q.replace("-", "").strip().strip('"').strip('"').strip()
    #             for q in queries.split("\n")
    #         ]
    #         queries = queries[: self.max_search_queries]

    #         # If seed is set, make queries deterministic
    #         if seed is not None:
    #             print(f"[Seed {seed}]: Queries before sorting: {queries}")
    #             queries = sorted(queries)
    #             # Maintain deterministic order while deduplicating
    #             unique_queries = []
    #             seen = set()
    #             for q in queries:
    #                 if q not in seen:
    #                     seen.add(q)
    #                     unique_queries.append(q)
    #             queries = unique_queries
    #             print(f"[Seed {seed}]: Queries after deduplication: {queries}")
    #         else:
    #             # Original behavior - non-deterministic deduplication
    #             queries = list(set(queries))

    #         # Perform search
    #         searched_results = self.retriever.retrieve(
    #             queries, exclude_urls=[ground_truth_url]
    #         )

    #         if seed is not None:
    #             print(
    #                 f"[Seed {seed}]: Search results first URLs: {[r.url for r in searched_results[:3]]}"
    #             )

    #             # First, sort snippets within each result deterministically.
    #             for result in searched_results:
    #                 if hasattr(result, "snippets") and result.snippets:
    #                     print(
    #                         f"[Seed {seed}]: Snippets before sorting for result {result.url}: {result.snippets}"
    #                     )
    #                     result.snippets = sorted(result.snippets)
    #                     print(
    #                         f"[Seed {seed}]: Snippets after sorting for result {result.url}: {result.snippets}"
    #                     )

    #             # Then sort search results by URL and the (now sorted) snippets.
    #             searched_results = sorted(
    #                 searched_results, key=lambda r: (r.url, tuple(r.snippets))
    #             )
    #             print(
    #                 f"[Seed {seed}]: Sorted search results URLs: {[r.url.strip() for r in searched_results]}"
    #             )

    #         # Process search results
    #         if len(searched_results) > 0:
    #             info = ""
    #             for n, r in enumerate(searched_results):
    #                 # No need to sort snippets again, they're already sorted if seed is provided
    #                 info += "\n".join(f"[{n + 1}]: {s}" for s in r.snippets[:1])
    #                 info += "\n\n"

    #             info = ArticleTextProcessing.limit_word_count_preserve_newline(
    #                 info, 1000
    #             )

    #             try:
    #                 # Use same seed context if provided
    #                 answer = self.answer_question(
    #                     topic=topic, conv=question, info=info
    #                 ).answer
    #                 answer = ArticleTextProcessing.remove_uncompleted_sentences_with_citations(
    #                     answer
    #                 )
    #             except Exception as e:
    #                 logging.error(f"Error occurs when generating answer: {e}")
    #                 answer = "Sorry, I cannot answer this question. Please ask another question."
    #         else:
    #             answer = "Sorry, I cannot find information for this question. Please ask another question."

    #     return dspy.Prediction(
    #         queries=queries, searched_results=searched_results, answer=answer
    #     )

    def forward_working_almost(self, topic: str, question: str, ground_truth_url: str):
        seed = getattr(self, "seed", None)
        print("Seed is ", seed)
        if hasattr(self.engine, "kwargs"):
            print("Seed of self.engine is ", self.engine.kwargs.get("seed"))

        # Use seed in context if provided
        context_kwargs = {"lm": self.engine, "show_guidelines": False}
        if seed is not None:
            context_kwargs["seed"] = seed

        with dspy.settings.context(**context_kwargs):
            # Generate queries
            queries = self.generate_queries(topic=topic, question=question).queries
            queries = [
                q.replace("-", "").strip().strip('"').strip('"').strip()
                for q in queries.split("\n")
            ]
            queries = queries[: self.max_search_queries]

            # If seed is set, make queries deterministic
            if seed is not None:
                print(f"[Seed {seed}]: Queries before sorting: {queries}")
                queries = sorted(queries)
                # Maintain deterministic order while deduplicating
                unique_queries = []
                seen = set()
                for q in queries:
                    if q not in seen:
                        seen.add(q)
                        unique_queries.append(q)
                queries = unique_queries
                print(f"[Seed {seed}]: Queries after deduplication: {queries}")
            else:
                # Original behavior - non-deterministic deduplication
                queries = list(set(queries))

            # Perform search
            searched_results = self.retriever.retrieve(
                queries, exclude_urls=[ground_truth_url]
            )

            if seed is not None:
                print(
                    f"[Seed {seed}]: Search results first URLs: {[r.url for r in searched_results[:3]]}"
                )

                # The key fix: Sort search results by URL first, then sort snippets within each result
                searched_results = sorted(
                    searched_results, key=lambda r: (r.url, tuple(r.snippets))
                )
                # searched_results = sorted(searched_results, key=lambda r: r.url)

                # # Pre-process all snippets deterministically
                # for result in searched_results:
                #     if hasattr(result, 'snippets') and result.snippets:
                #         # Sort the snippets within each result
                #         print(f"[Seed {seed}]: Snippets before sorting: {result.snippets}")
                #         result.snippets = sorted(result.snippets)
                #         print(f"[Seed {seed}]: Snippets after sorting: {result.snippets}")

            # Process search results
            if len(searched_results) > 0:
                info = ""
                for n, r in enumerate(searched_results):
                    # No need to sort snippets again, they're already sorted if seed is provided
                    info += "\n".join(f"[{n + 1}]: {s}" for s in r.snippets[:1])
                    info += "\n\n"

                info = ArticleTextProcessing.limit_word_count_preserve_newline(
                    info, 1000
                )

                try:
                    # Use same seed context if provided
                    answer = self.answer_question(
                        topic=topic, conv=question, info=info
                    ).answer
                    answer = ArticleTextProcessing.remove_uncompleted_sentences_with_citations(
                        answer
                    )
                except Exception as e:
                    logging.error(f"Error occurs when generating answer: {e}")
                    answer = "Sorry, I cannot answer this question. Please ask another question."
            else:
                answer = "Sorry, I cannot find information for this question. Please ask another question."

        return dspy.Prediction(
            queries=queries, searched_results=searched_results, answer=answer
        )

    def forward_(self, topic: str, question: str, ground_truth_url: str):
        seed = getattr(self, "seed", None)
        print("Seed is ", seed)
        print("Seed of self.engine is ", self.engine.kwargs["seed"])
        with dspy.settings.context(
            lm=self.engine, show_guidelines=False, seed=self.seed
        ):
            # Identify: Break down question into queries.
            queries = self.generate_queries(topic=topic, question=question).queries
            queries = [
                q.replace("-", "").strip().strip('"').strip('"').strip()
                for q in queries.split("\n")
            ]
            queries = queries[: self.max_search_queries]
            # print(f"🔹 Generated queries: {queries}")
            # In TopicExpert.forward

            print(f"[Seed {self.seed}]: Queries before sorting: {queries}")
            # Search
            if self.seed is not None:
                queries = sorted(queries)
                # searched_results = self.retriever.retrieve(queries, exclude_urls=[ground_truth_url])
                unique_queries = []
                for q in queries:
                    if q not in unique_queries:
                        unique_queries.append(q)
                searched_results = self.retriever.retrieve(
                    unique_queries, exclude_urls=[ground_truth_url]
                )
                print(
                    f"[Seed {self.seed}]: Queries after deduplication: {unique_queries}"
                )
            else:
                searched_results: List[Information] = self.retriever.retrieve(
                    list(set(queries)), exclude_urls=[ground_truth_url]
                )
            print(
                f"[Seed {self.seed}]: Search results first URLs: {[r.url for r in searched_results[:3]]}"
            )

            if len(searched_results) > 0:
                # Ensure deterministic order of search results when seed is set
                if self.seed is not None:
                    # Sort search results by URL for complete determinism
                    searched_results = sorted(searched_results, key=lambda r: r.url)

                info = ""
                for n, r in enumerate(searched_results):
                    # Ensure deterministic order of snippets when seed is set
                    if self.seed is not None and r.snippets:
                        print(
                            f"[Seed {self.seed}]: Snippets before sorting: {r.snippets}"
                        )
                        snippets = sorted(r.snippets)
                        print(f"[Seed {self.seed}]: Snippets after sorting: {snippets}")
                    else:
                        snippets = r.snippets if hasattr(r, "snippets") else []

                    # Use only first snippet from each result
                    if snippets:
                        info += f"[{n + 1}]: {snippets[0]}\n\n"

                info = ArticleTextProcessing.limit_word_count_preserve_newline(
                    info, 1000
                )

                try:
                    # Generate answer with same seed context
                    with dspy.settings.context(lm=self.engine, seed=self.seed):
                        answer = self.answer_question(
                            topic=topic, conv=question, info=info
                        ).answer

                    answer = ArticleTextProcessing.remove_uncompleted_sentences_with_citations(
                        answer
                    )

                    # Optional: Log hash for debugging
                    if self.seed is not None:
                        import hashlib

                        answer_hash = hashlib.md5(answer.encode()).hexdigest()
                        print(f"[Seed {self.seed}]: Answer hash: {answer_hash}")

                except Exception as e:
                    logging.error(f"Error occurs when generating answer: {e}")
                    answer = "Sorry, I cannot answer this question. Please ask another question."
            else:
                answer = "Sorry, I cannot find information for this question. Please ask another question."

            # Ensure deterministic return order for queries (they might have been modified)
            # if self.seed is not None:
            #     # Sort queries one last time to guarantee determinism
            #     queries = sorted(queries)

            return dspy.Prediction(
                queries=queries, searched_results=searched_results, answer=answer
            )


class StormKnowledgeCurationModule(KnowledgeCurationModule):
    """
    The interface for knowledge curation stage. Given topic, return collected information.
    """

    def __init__(
        self,
        retriever: Retriever,
        persona_generator: Optional[StormPersonaGenerator],
        conv_simulator_lm: Union[dspy.LM],
        question_asker_lm: Union[dspy.LM],
        max_search_queries_per_turn: int,
        search_top_k: int,
        max_conv_turn: int,
        max_thread_num: int,
        embedding_model: str,
        seed: Optional[int] = None,
    ):
        """
        Store args and finish initialization.
        """
        super().__init__(retriever)
        self.persona_generator = persona_generator
        self.conv_simulator_lm = conv_simulator_lm
        self.question_asker_lm = question_asker_lm
        self.max_search_queries_per_turn = max_search_queries_per_turn
        self.max_conv_turn = max_conv_turn
        self.search_top_k = search_top_k
        self.max_thread_num = max_thread_num
        self.conv_simulator = ConvSimulator(
            topic_expert_engine=conv_simulator_lm,
            question_asker_engine=question_asker_lm,
            retriever=retriever,
            max_search_queries_per_turn=max_search_queries_per_turn,
            search_top_k=search_top_k,
            max_turn=max_conv_turn,
            seed=seed,
        )
        self.embedding_model = embedding_model
        self.seed = seed

    def _get_considered_personas(self, topic: str, max_num_persona) -> List[str]:
        return self.persona_generator.generate_persona(
            topic=topic, max_num_persona=max_num_persona
        )

    def _run_conversation(
        self,
        conv_simulator,
        topic,
        ground_truth_url,
        considered_personas,
        callback_handler: BaseCallbackHandler,
    ) -> List[Tuple[str, List[DialogueTurn]]]:
        """
        Executes multiple conversation simulations concurrently, each with a different persona,
        and collects their dialog histories. The dialog history of each conversation is cleaned
        up before being stored.

        Parameters:
            conv_simulator (callable): The function to simulate conversations. It must accept four
                parameters: `topic`, `ground_truth_url`, `persona`, and `callback_handler`, and return
                an object that has a `dlg_history` attribute.
            topic (str): The topic of conversation for the simulations.
            ground_truth_url (str): The URL to the ground truth data related to the conversation topic.
            considered_personas (list): A list of personas under which the conversation simulations
                will be conducted. Each persona is passed to `conv_simulator` individually.
            callback_handler (callable): A callback function that is passed to `conv_simulator`. It
                should handle any callbacks or events during the simulation.

        Returns:
            list of tuples: A list where each tuple contains a persona and its corresponding cleaned
            dialog history (`dlg_history`) from the conversation simulation.
        """

        conversations = []

        if self.seed is not None:
            print(f"🟢 Deterministic Mode (seed={self.seed}): Running sequentially.")
            # Sort personas so we get stable iteration order
            sorted_personas = (
                sorted(considered_personas) if considered_personas else [""]
            )

            print("sorted_personas: ", sorted_personas)

            for persona in sorted_personas:
                # print(f"Processing persona: {persona}")
                conv = conv_simulator(
                    topic=topic,
                    ground_truth_url=ground_truth_url,
                    persona=persona,
                    callback_handler=callback_handler,
                )
                conversations.append(
                    (persona, ArticleTextProcessing.clean_up_citation(conv).dlg_history)
                )
        else:
            print("🔵 Non-deterministic Mode: Running concurrently.")

            def run_conv(persona):
                return conv_simulator(
                    topic=topic,
                    ground_truth_url=ground_truth_url,
                    persona=persona,
                    callback_handler=callback_handler,
                )

            max_workers = min(self.max_thread_num, len(considered_personas))

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                future_to_persona = {
                    executor.submit(run_conv, persona): persona
                    for persona in considered_personas
                }

                if streamlit_connection:
                    # Ensure the logging context is correct when connecting with Streamlit frontend.
                    for t in executor._threads:
                        add_script_run_ctx(t)

                for future in as_completed(future_to_persona):
                    persona = future_to_persona[future]
                    conv = future.result()
                    conversations.append(
                        (
                            persona,
                            ArticleTextProcessing.clean_up_citation(conv).dlg_history,
                        )
                    )

        return conversations

    def research(
        self,
        topic: str,
        ground_truth_url: str,
        callback_handler: BaseCallbackHandler,
        max_perspective: int = 0,
        disable_perspective: bool = True,
        return_conversation_log=False,
    ) -> Union[StormInformationTable, Tuple[StormInformationTable, Dict]]:
        """
        Curate information and knowledge for the given topic

        Args:
            topic: topic of interest in natural language.

        Returns:
            collected_information: collected information in InformationTable type.
        """

        # identify personas
        callback_handler.on_identify_perspective_start()
        considered_personas = []
        if disable_perspective:
            considered_personas = [""]
        else:
            considered_personas = self._get_considered_personas(
                topic=topic, max_num_persona=max_perspective
            )
        callback_handler.on_identify_perspective_end(perspectives=considered_personas)

        # run conversation
        callback_handler.on_information_gathering_start()
        conversations = self._run_conversation(
            conv_simulator=self.conv_simulator,
            topic=topic,
            ground_truth_url=ground_truth_url,
            considered_personas=considered_personas,
            callback_handler=callback_handler,
        )

        information_table = StormInformationTable(conversations, self.embedding_model)
        callback_handler.on_information_gathering_end()
        if return_conversation_log:
            return information_table, StormInformationTable.construct_log_dict(
                conversations
            )
        return information_table
