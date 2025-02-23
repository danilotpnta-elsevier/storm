# lm.py
import logging
import threading
import dspy
from typing import Optional, Literal

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
            if hasattr(self, 'history') and self.history:
                last_call = self.history[-1]
                usage_data = last_call.get('usage')
                
                if usage_data:
                    with self._token_usage_lock:
                        self.prompt_tokens += usage_data.get('prompt_tokens', 0)
                        self.completion_tokens += usage_data.get('completion_tokens', 0)
                        logging.debug(f"Updated tokens - Prompt: {self.prompt_tokens}, Completion: {self.completion_tokens}")
                        return
            
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
            logging.error(f"Response type: {type(response)}")
            if isinstance(response, dict):
                logging.error(f"Response keys: {list(response.keys())}")

    def __call__(self, *args, **kwargs):
        """Override __call__ to ensure we capture usage from the history."""
        result = super().__call__(*args, **kwargs)
        
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
            self.prompt_tokens = 0
            self.completion_tokens = 0
            return usage