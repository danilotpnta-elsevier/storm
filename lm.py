# lm.py
import os
import logging
import threading
import dspy
from typing import Optional, Literal, Any

import logging

logging.getLogger("LiteLLM").setLevel(logging.WARNING)


class LLM(dspy.LM):
    """Language class Manager to initialize Azure or Bedrock models"""

    BEDROCK_MODEL_CONFIGS = {
        "llama-3-1-8B": {
            "model": "meta.llama3-1-8b-instruct-v1:0",
            "aws_region_name": "us-west-2",
        },
        "llama-3-3-70B": {
            "model": "us.meta.llama3-3-70b-instruct-v1:0",
            "aws_region_name": "us-west-2",
        },
        "llama-3-1-70B": {
            "model": "meta.llama3-1-70b-instruct-v1:0",
            "aws_region_name": "us-west-2",
        },
        "llama-3-70B": {
            "model": "meta.llama3-70b-instruct-v1:0",
            "aws_region_name": "us-west-2",
        },
        "mistral-7b-v2": {
            "model": "mistral.mistral-7b-instruct-v0:2",
            "aws_region_name": "us-west-2",
        },
        "claude-3-5-sonnet": {
            "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "aws_region_name": "us-west-2",
        },
        "claude-3-7-sonnet": {
            "model": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            "aws_region_name": "us-west-2",
        },
    }

    AZURE_MODELS = [
        "gpt-4",
        "gpt-4o-mini",
        "gpt-3.5-turbo",
    ]

    def __init__(
        self,
        model: str,
        provider: Optional[Literal["azure", "bedrock"]] = None,
        # Azure-specific parameters
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        api_key: Optional[str] = None,
        deployment_name: Optional[str] = None,
        # Bedrock-specific parameters
        aws_region_name: Optional[str] = "us-west-2",
        aws_profile_name: Optional[str] = "danilotpnta-elsevier",
        # Common parameters
        model_type: Literal["chat", "text"] = "chat",
        **kwargs,
    ):
        if model is None:
            raise ValueError("Model must be specified")
        if provider is None:
            provider = self._detect_provider(model)

        self.provider = provider
        self._token_usage_lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0

        if provider == "azure":
            api_key = api_key or os.getenv("AZURE_API_KEY")
            api_base = api_base or os.getenv("AZURE_API_BASE")
            api_version = api_version or os.getenv("AZURE_API_VERSION")

            model_identifier = (
                f"azure/{deployment_name}" if deployment_name else f"azure/{model}"
            )
            self.model_name = deployment_name or model

            azure_defaults = {
                "top_p": 0.9,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "n": 1,
            }
            kwargs = {**azure_defaults, **kwargs}

            super().__init__(
                model=model_identifier,
                api_base=api_base,
                api_version=api_version,
                api_key=api_key,
                model_type=model_type,
                **kwargs,
            )

        elif provider == "bedrock":

            if model in self.BEDROCK_MODEL_CONFIGS:
                config = self.BEDROCK_MODEL_CONFIGS[model]
                bedrock_model = config["model"]
                aws_region_name = config["aws_region_name"]
            else:
                bedrock_model = model

            model_identifier = f"bedrock/{bedrock_model}"
            self.model_name = bedrock_model

            bedrock_defaults = {
                "top_p": 0.9,
                "n": 1,
            }
            kwargs = {**bedrock_defaults, **kwargs}

            super().__init__(
                model=model_identifier,
                aws_region_name=aws_region_name,
                aws_profile_name=aws_profile_name,
                model_type=model_type,
                **kwargs,
            )

        else:
            raise ValueError(
                f"Provider {provider} not supported. Choose 'azure' or 'bedrock'."
            )

    def _detect_provider(self, model: str) -> str:
        """
        Automatically detect the provider based on the model name.

        Args:
            model: The model name/identifier

        Returns:
            str: Either "azure" or "bedrock"
        """
        for azure_prefix in self.AZURE_MODELS:
            if model.startswith(azure_prefix):
                return "azure"

        if model in self.BEDROCK_MODEL_CONFIGS:
            return "bedrock"

    def log_usage(self, response):
        """Log the total tokens from the API response."""
        try:
            if hasattr(self, "history") and self.history:
                last_call = self.history[-1]
                usage_data = last_call.get("usage")

                if usage_data:
                    with self._token_usage_lock:
                        self.prompt_tokens += usage_data.get("prompt_tokens", 0)
                        self.completion_tokens += usage_data.get("completion_tokens", 0)
                        logging.debug(
                            f"Updated tokens - Prompt: {self.prompt_tokens}, Completion: {self.completion_tokens}"
                        )
                        return

            if isinstance(response, dict):
                usage_data = response.get("usage")
            else:
                usage_data = getattr(response, "usage", None)

            if usage_data:
                with self._token_usage_lock:
                    self.prompt_tokens += usage_data.get("prompt_tokens", 0)
                    self.completion_tokens += usage_data.get("completion_tokens", 0)
                    logging.debug(
                        f"Updated tokens from response - Prompt: {self.prompt_tokens}, Completion: {self.completion_tokens}"
                    )

        except Exception as e:
            logging.error(f"Error in log_usage: {str(e)}")
            logging.error(f"Response type: {type(response)}")
            if isinstance(response, dict):
                logging.error(f"Response keys: {list(response.keys())}")

    def __call__(self, *args, **kwargs):
        """Override __call__ to ensure we capture usage from the history."""
        result = super().__call__(*args, **kwargs)

        if self.history and self.history[-1].get("usage"):
            usage_data = self.history[-1]["usage"]
            with self._token_usage_lock:
                self.prompt_tokens += usage_data.get("prompt_tokens", 0)
                self.completion_tokens += usage_data.get("completion_tokens", 0)

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
        model_identifier = (
            f"azure/{deployment_name}" if deployment_name else f"azure/{model}"
        )
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
            if hasattr(self, "history") and self.history:
                last_call = self.history[-1]
                usage_data = last_call.get("usage")

                if usage_data:
                    with self._token_usage_lock:
                        self.prompt_tokens += usage_data.get("prompt_tokens", 0)
                        self.completion_tokens += usage_data.get("completion_tokens", 0)
                        logging.debug(
                            f"Updated tokens - Prompt: {self.prompt_tokens}, Completion: {self.completion_tokens}"
                        )
                        return

            if isinstance(response, dict):
                usage_data = response.get("usage")
            else:
                usage_data = getattr(response, "usage", None)

            if usage_data:
                with self._token_usage_lock:
                    self.prompt_tokens += usage_data.get("prompt_tokens", 0)
                    self.completion_tokens += usage_data.get("completion_tokens", 0)
                    logging.debug(
                        f"Updated tokens from response - Prompt: {self.prompt_tokens}, Completion: {self.completion_tokens}"
                    )

        except Exception as e:
            logging.error(f"Error in log_usage: {str(e)}")
            logging.error(f"Response type: {type(response)}")
            if isinstance(response, dict):
                logging.error(f"Response keys: {list(response.keys())}")

    def __call__(self, *args, **kwargs):
        """Override __call__ to ensure we capture usage from the history."""
        result = super().__call__(*args, **kwargs)

        if self.history and self.history[-1].get("usage"):
            usage_data = self.history[-1]["usage"]
            with self._token_usage_lock:
                self.prompt_tokens += usage_data.get("prompt_tokens", 0)
                self.completion_tokens += usage_data.get("completion_tokens", 0)

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
