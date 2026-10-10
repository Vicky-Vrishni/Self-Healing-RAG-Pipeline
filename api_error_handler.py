
import logging

import groq
from groq import Groq

from retrieval_error_handling import (
    RAGPipelineError,
    log_pipeline_error,
)

logger = logging.getLogger(__name__)

# Type of Error raised When API doesn't connect.
class APIError(RAGPipelineError):
    """Base exception for external API failures."""


class APIAuthenticationError(APIError):
    """Raised when API authentication fails."""


class APIRateLimitError(APIError):
    """Raised when the API rate limit is exceeded."""


class APITimeoutError(APIError):
    """Raised when the API request times out."""


class APIConnectionError(APIError):
    """Raised when the API cannot be reached."""


def call_groq_api(client: Groq, **kwargs) -> str:
    """
    Call Groq and convert provider errors into
    controlled application exceptions.
    """
    try:
        response = client.chat.completions.create(**kwargs)

        answer = response.choices[0].message.content

        if not answer:
            raise APIError("The API returned an empty response.")

        return answer

    except groq.AuthenticationError as error:
        log_pipeline_error("api_authentication", error)
        raise APIAuthenticationError(
            "API authentication failed. Check server configuration."
        ) from error

    except groq.RateLimitError as error:
        log_pipeline_error("api_rate_limit", error)
        raise APIRateLimitError(
            "The API rate limit was reached. Please try again later."
        ) from error

    except groq.APITimeoutError as error:
        log_pipeline_error("api_timeout", error)
        raise APITimeoutError(
            "The API request timed out. Please try again."
        ) from error

    except groq.APIConnectionError as error:
        log_pipeline_error("api_connection", error)
        raise APIConnectionError(
            "The API service could not be reached."
        ) from error

    except groq.APIError as error:
        log_pipeline_error("api_request", error)
        raise APIError(
            "The language model service encountered an error."
        ) from error
