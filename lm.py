import logging
import os
import random
import threading
from typing import Optional, Literal, Any

# import backoff
import dspy

# import requests
# from dsp import ERRORS, backoff_hdlr, giveup_hdlr
# from dsp.modules.hf import openai_to_hf
# from dsp.modules.hf_client import send_hftgi_request_v01_wrapped
# from openai import OpenAI
# from transformers import AutoTokenizer


class OpenAIModel(dspy.LM):
    """A wrapper class for dspy.OpenAI."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        model_type: Literal["chat", "text"] = None,
        **kwargs,
    ):
        super().__init__(model=model, api_key=api_key, model_type=model_type, **kwargs)
        self._token_usage_lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def log_usage(self, response):
        """Log the total tokens from the OpenAI API response."""
        usage_data = response.get("usage")
        if usage_data:
            with self._token_usage_lock:
                self.prompt_tokens += usage_data.get("prompt_tokens", 0)
                self.completion_tokens += usage_data.get("completion_tokens", 0)

    def get_usage_and_reset(self):
        """Get the total tokens used and reset the token usage."""
        usage = {
            self.kwargs.get("model")
            or self.kwargs.get("engine"): {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
            }
        }
        self.prompt_tokens = 0
        self.completion_tokens = 0

        return usage

    def __call__(
        self,
        prompt: str,
        only_completed: bool = True,
        return_sorted: bool = False,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Copied from dspy/dsp/modules/gpt3.py with the addition of tracking token usage."""

        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        # if kwargs.get("n", 1) > 1:
        #     if self.model_type == "chat":
        #         kwargs = {**kwargs}
        #     else:
        #         kwargs = {**kwargs, "logprobs": 5}

        response = self.request(prompt, **kwargs)

        # Log the token usage from the OpenAI API response.
        self.log_usage(response)

        choices = response["choices"]

        completed_choices = [c for c in choices if c["finish_reason"] != "length"]

        if only_completed and len(completed_choices):
            choices = completed_choices

        completions = [self._get_choice_text(c) for c in choices]
        if return_sorted and kwargs.get("n", 1) > 1:
            scored_completions = []

            for c in choices:
                tokens, logprobs = (
                    c["logprobs"]["tokens"],
                    c["logprobs"]["token_logprobs"],
                )

                if "<|endoftext|>" in tokens:
                    index = tokens.index("<|endoftext|>") + 1
                    tokens, logprobs = tokens[:index], logprobs[:index]

                avglog = sum(logprobs) / len(logprobs)
                scored_completions.append((avglog, self._get_choice_text(c)))

            scored_completions = sorted(scored_completions, reverse=True)
            completions = [c for _, c in scored_completions]

        return completions


import logging
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

class AzureOpenAIModel_(dspy.LM):
    """A wrapper class for dspy.AzureOpenAI."""

    def __init__(
        self,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        deployment_name: Optional[str] = None,
        model_type: Literal["chat", "text"] = "chat",
        **kwargs,
    ):  
        model_identifier = f"azure/{deployment_name}" if deployment_name else f"azure/{model}"
        super().__init__(
            model=model_identifier,
            api_base=api_base,
            api_version=api_version,
            api_key=api_key,
            model_type=model_type,
            **kwargs,
        )
        self._token_usage_lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def log_usage(self, response):
        """Log the total tokens from the OpenAI API response.
        Override log_usage() in dspy.AzureOpenAI for tracking accumulated token usage.
        """
        usage_data = response.get("usage")
        if usage_data:
            with self._token_usage_lock:
                self.prompt_tokens += usage_data.get("prompt_tokens", 0)
                self.completion_tokens += usage_data.get("completion_tokens", 0)

    def get_usage_and_reset(self):
        """Get the total tokens used and reset the token usage."""
        usage = {
            self.kwargs.get("model")
            or self.kwargs.get("engine"): {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
            }
        }
        self.prompt_tokens = 0
        self.completion_tokens = 0

        return usage

class AzureOpenAIModel(dspy.LM):
    def __init__(
        self,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        deployment_name: Optional[str] = None,
        model_type: Literal["chat", "text"] = "chat",
        **kwargs,
    ):  
        model_identifier = f"azure/{deployment_name}" if deployment_name else f"azure/{model}"
        super().__init__(
            model=model_identifier,
            api_base=api_base,
            api_version=api_version,
            api_key=api_key,
            model_type=model_type,
            **kwargs,
        )
        self._token_usage_lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.model_name = deployment_name or model

    def log_usage(self, response):
        """Log the total tokens from the OpenAI API response."""
        try:
            # Get the last history entry directly from self.history
            if hasattr(self, 'history') and self.history:
                last_call = self.history[-1]
                usage_data = last_call.get('usage')
                
                if usage_data:
                    with self._token_usage_lock:
                        self.prompt_tokens += usage_data.get('prompt_tokens', 0)
                        self.completion_tokens += usage_data.get('completion_tokens', 0)
                        logging.debug(f"Updated tokens - Prompt: {self.prompt_tokens}, Completion: {self.completion_tokens}")
                        return
            
            # Fallback: try to get usage from the response itself
            if isinstance(response, dict):
                usage_data = response.get('usage')
            else:
                usage_data = getattr(response, 'usage', None)
            
            if usage_data:
                with self._token_usage_lock:
                    self.prompt_tokens += usage_data.get('prompt_tokens', 0)
                    self.completion_tokens += usage_data.get('completion_tokens', 0)
                    logging.debug(f"Updated tokens from response - Prompt: {self.prompt_tokens}, Completion: {self.completion_tokens}")
            
        except Exception as e:
            logging.error(f"Error in log_usage: {str(e)}")
            # Log the response type and structure for debugging
            logging.error(f"Response type: {type(response)}")
            if isinstance(response, dict):
                logging.error(f"Response keys: {list(response.keys())}")

    def __call__(self, *args, **kwargs):
        """Override __call__ to ensure we capture usage from the history."""
        result = super().__call__(*args, **kwargs)
        
        # After the call completes, check history for usage
        if self.history and self.history[-1].get('usage'):
            usage_data = self.history[-1]['usage']
            with self._token_usage_lock:
                self.prompt_tokens += usage_data.get('prompt_tokens', 0)
                self.completion_tokens += usage_data.get('completion_tokens', 0)
        
        return result

    def get_usage_and_reset(self):
        """Get the total tokens used and reset the token usage."""
        with self._token_usage_lock:
            usage = {
                self.model_name: {
                    "prompt_tokens": self.prompt_tokens,
                    "completion_tokens": self.completion_tokens,
                }
            }
            # Reset counters
            self.prompt_tokens = 0
            self.completion_tokens = 0
            return usage