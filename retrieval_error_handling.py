import logging

logger = logging.getLogger(__name__)


class RAGPipelineError(Exception):
    """Base exception for controlled RAG pipeline errors."""


class RetrievalError(RAGPipelineError):
    """Raised when document retrieval fails."""


class GenerationError(RAGPipelineError):
    """Raised when answer generation fails."""


class CriticError(RAGPipelineError):
    """Raised when answer evaluation fails."""


def log_pipeline_error(
    stage: str,
    error: Exception
) -> None:
    """Log an internal error without exposing it to users."""
    logger.exception(
        "RAG pipeline failed at stage: %s (%s)",
        stage,
        type(error).__name__
    )