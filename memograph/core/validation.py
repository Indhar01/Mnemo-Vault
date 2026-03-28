"""Input validation and error handling for MemoGraph.

This module provides comprehensive input validation with:
- Type checking and range validation
- Path sanitization and security checks
- Query validation with helpful suggestions
- Custom exception classes with context
- Detailed error messages with fix suggestions

Example:
    >>> from memograph.core.validation import validate_query, validate_path
    >>> validate_query("python tips")  # OK
    >>> validate_query("")  # Raises ValidationError with suggestion
    >>> validate_path("/safe/path/file.md")  # OK
    >>> validate_path("../../../etc/passwd")  # Raises SecurityError
"""

import logging
import re
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger("memograph.validation")


# ============================================================================
# Custom Exception Classes
# ============================================================================


class MemoGraphError(Exception):
    """Base exception for all MemoGraph errors."""

    def __init__(
        self,
        message: str,
        suggestion: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ):
        """Initialize error with message, suggestion, and context.

        Args:
            message: Error message describing what went wrong
            suggestion: Helpful suggestion on how to fix the error
            context: Additional context information (dict)
        """
        self.message = message
        self.suggestion = suggestion
        self.context = context or {}

        # Build full error message
        full_message = message
        if suggestion:
            full_message += f"\n\nSuggestion: {suggestion}"
        if context:
            context_str = "\n".join(f"  {k}: {v}" for k, v in context.items())
            full_message += f"\n\nContext:\n{context_str}"

        super().__init__(full_message)


class ValidationError(MemoGraphError):
    """Raised when input validation fails."""

    pass


class SecurityError(MemoGraphError):
    """Raised when a security check fails (e.g., path traversal)."""

    pass


class ConfigurationError(MemoGraphError):
    """Raised when configuration is invalid or missing."""

    pass


class GraphError(MemoGraphError):
    """Raised when graph operations fail."""

    pass


class EmbeddingError(MemoGraphError):
    """Raised when embedding operations fail."""

    pass


class RetrievalError(MemoGraphError):
    """Raised when retrieval operations fail."""

    pass


# ============================================================================
# Validation Functions
# ============================================================================


def validate_query(
    query: str, min_length: int = 1, max_length: int = 1000, allow_empty: bool = False
) -> str:
    """Validate search query string.

    Args:
        query: Query string to validate
        min_length: Minimum query length
        max_length: Maximum query length
        allow_empty: Whether to allow empty queries

    Returns:
        Validated and normalized query string

    Raises:
        ValidationError: If query is invalid

    Example:
        >>> validate_query("python tips")
        'python tips'
        >>> validate_query("")
        ValidationError: Query cannot be empty
    """
    # Type check
    if not isinstance(query, str):
        raise ValidationError(
            f"Query must be a string, got {type(query).__name__}",
            suggestion="Pass a string query like 'python tips'",
            context={"received_type": type(query).__name__},
        )

    # Empty check
    if not query.strip():
        if not allow_empty:
            raise ValidationError(
                "Query cannot be empty or whitespace-only",
                suggestion="Provide a meaningful search query (e.g., 'python tips', 'machine learning')",
                context={"query": repr(query)},
            )
        return ""

    # Length checks
    query = query.strip()

    if len(query) < min_length:
        raise ValidationError(
            f"Query too short: {len(query)} characters (minimum: {min_length})",
            suggestion=f"Provide a query with at least {min_length} character(s)",
            context={"query": query, "length": len(query), "min_length": min_length},
        )

    if len(query) > max_length:
        raise ValidationError(
            f"Query too long: {len(query)} characters (maximum: {max_length})",
            suggestion=f"Shorten your query to {max_length} characters or less",
            context={"query": query[:100] + "...", "length": len(query), "max_length": max_length},
        )

    logger.debug(f"Validated query: {query}")
    return query


def validate_path(
    path: Union[str, Path],
    must_exist: bool = False,
    must_be_file: bool = False,
    must_be_dir: bool = False,
    allowed_extensions: Optional[list[str]] = None,
    base_path: Optional[Path] = None,
) -> Path:
    """Validate and sanitize file/directory path.

    Args:
        path: Path to validate
        must_exist: Whether path must exist
        must_be_file: Whether path must be a file
        must_be_dir: Whether path must be a directory
        allowed_extensions: List of allowed file extensions (e.g., ['.md', '.txt'])
        base_path: Base path for security checks (prevents path traversal)

    Returns:
        Validated Path object

    Raises:
        ValidationError: If path is invalid
        SecurityError: If path fails security checks

    Example:
        >>> validate_path("./vault/memory.md", must_exist=True, must_be_file=True)
        Path('./vault/memory.md')
        >>> validate_path("../../../etc/passwd", base_path=Path("./vault"))
        SecurityError: Path traversal detected
    """
    # Type check
    if not isinstance(path, (str, Path)):
        raise ValidationError(
            f"Path must be a string or Path object, got {type(path).__name__}",
            suggestion="Pass a valid file path as a string or Path object",
            context={"received_type": type(path).__name__},
        )

    # Convert to Path
    path = Path(path)

    # Security check: Path traversal
    if base_path:
        try:
            resolved_path = path.resolve()
            resolved_base = base_path.resolve()

            # Check if path is within base_path
            if not str(resolved_path).startswith(str(resolved_base)):
                raise SecurityError(
                    "Path traversal detected: Path is outside allowed directory",
                    suggestion=f"Use paths within {base_path}",
                    context={
                        "path": str(path),
                        "resolved": str(resolved_path),
                        "base_path": str(base_path),
                    },
                )
        except Exception as e:
            if isinstance(e, SecurityError):
                raise
            logger.warning(f"Failed to resolve path for security check: {e}")

    # Existence checks
    if must_exist and not path.exists():
        raise ValidationError(
            f"Path does not exist: {path}",
            suggestion="Check the path and ensure the file/directory exists",
            context={"path": str(path)},
        )

    if path.exists():
        if must_be_file and not path.is_file():
            raise ValidationError(
                f"Path is not a file: {path}",
                suggestion="Provide a path to a file, not a directory",
                context={"path": str(path), "is_dir": path.is_dir()},
            )

        if must_be_dir and not path.is_dir():
            raise ValidationError(
                f"Path is not a directory: {path}",
                suggestion="Provide a path to a directory, not a file",
                context={"path": str(path), "is_file": path.is_file()},
            )

    # Extension check
    if allowed_extensions and path.suffix:
        if path.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
            raise ValidationError(
                f"Invalid file extension: {path.suffix}",
                suggestion=f"Use one of these extensions: {', '.join(allowed_extensions)}",
                context={
                    "path": str(path),
                    "extension": path.suffix,
                    "allowed": allowed_extensions,
                },
            )

    logger.debug(f"Validated path: {path}")
    return path


def validate_tags(
    tags: Union[str, list[str]], max_tags: int = 20, max_tag_length: int = 50
) -> list[str]:
    """Validate and normalize tags.

    Args:
        tags: Single tag string or list of tags
        max_tags: Maximum number of tags allowed
        max_tag_length: Maximum length per tag

    Returns:
        List of validated and normalized tags

    Raises:
        ValidationError: If tags are invalid

    Example:
        >>> validate_tags("python")
        ['python']
        >>> validate_tags(["python", "machine-learning", "AI"])
        ['python', 'machine-learning', 'ai']
    """
    # Convert single tag to list
    if isinstance(tags, str):
        tags = [tags]

    # Type check
    if not isinstance(tags, list):
        raise ValidationError(
            f"Tags must be a string or list of strings, got {type(tags).__name__}",
            suggestion="Pass tags as a string or list: 'python' or ['python', 'ml']",
            context={"received_type": type(tags).__name__},
        )

    # Validate each tag
    validated_tags = []
    for tag in tags:
        if not isinstance(tag, str):
            raise ValidationError(
                f"Each tag must be a string, got {type(tag).__name__}",
                suggestion="Ensure all tags are strings",
                context={"tag": tag, "type": type(tag).__name__},
            )

        # Normalize: strip, lowercase, remove special chars
        tag = tag.strip().lower()
        tag = re.sub(r"[^\w\-]", "", tag)

        if not tag:
            continue  # Skip empty tags

        if len(tag) > max_tag_length:
            raise ValidationError(
                f"Tag too long: '{tag}' ({len(tag)} characters, max: {max_tag_length})",
                suggestion=f"Shorten tag to {max_tag_length} characters or less",
                context={"tag": tag, "length": len(tag)},
            )

        validated_tags.append(tag)

    # Remove duplicates while preserving order
    validated_tags = list(dict.fromkeys(validated_tags))

    # Check max tags
    if len(validated_tags) > max_tags:
        raise ValidationError(
            f"Too many tags: {len(validated_tags)} (maximum: {max_tags})",
            suggestion=f"Reduce to {max_tags} most relevant tags",
            context={"count": len(validated_tags), "max": max_tags},
        )

    logger.debug(f"Validated tags: {validated_tags}")
    return validated_tags


def validate_salience(salience: float) -> float:
    """Validate salience score.

    Args:
        salience: Salience score to validate

    Returns:
        Validated salience score

    Raises:
        ValidationError: If salience is invalid

    Example:
        >>> validate_salience(0.8)
        0.8
        >>> validate_salience(1.5)
        ValidationError: Salience must be between 0.0 and 1.0
    """
    # Type check
    if not isinstance(salience, (int, float)):
        raise ValidationError(
            f"Salience must be a number, got {type(salience).__name__}",
            suggestion="Provide a salience score between 0.0 and 1.0",
            context={"received_type": type(salience).__name__},
        )

    # Range check
    if not 0.0 <= salience <= 1.0:
        raise ValidationError(
            f"Salience must be between 0.0 and 1.0, got {salience}",
            suggestion="Use a value between 0.0 (low importance) and 1.0 (high importance)",
            context={"salience": salience},
        )

    return float(salience)


def validate_depth(depth: int, max_depth: int = 10) -> int:
    """Validate graph traversal depth.

    Args:
        depth: Depth to validate
        max_depth: Maximum allowed depth

    Returns:
        Validated depth

    Raises:
        ValidationError: If depth is invalid

    Example:
        >>> validate_depth(2)
        2
        >>> validate_depth(-1)
        ValidationError: Depth must be positive
    """
    # Type check
    if not isinstance(depth, int):
        raise ValidationError(
            f"Depth must be an integer, got {type(depth).__name__}",
            suggestion="Provide a positive integer for graph traversal depth",
            context={"received_type": type(depth).__name__},
        )

    # Range checks
    if depth < 0:
        raise ValidationError(
            f"Depth must be non-negative, got {depth}",
            suggestion="Use a positive integer (e.g., 1, 2, 3) or 0 for no traversal",
            context={"depth": depth},
        )

    if depth > max_depth:
        raise ValidationError(
            f"Depth too large: {depth} (maximum: {max_depth})",
            suggestion=f"Use a depth of {max_depth} or less to avoid performance issues",
            context={"depth": depth, "max_depth": max_depth},
        )

    return depth


def validate_top_k(top_k: int, max_top_k: int = 100) -> int:
    """Validate top-k parameter for retrieval.

    Args:
        top_k: Number of results to return
        max_top_k: Maximum allowed top-k

    Returns:
        Validated top-k

    Raises:
        ValidationError: If top-k is invalid

    Example:
        >>> validate_top_k(10)
        10
        >>> validate_top_k(0)
        ValidationError: top_k must be positive
    """
    # Type check
    if not isinstance(top_k, int):
        raise ValidationError(
            f"top_k must be an integer, got {type(top_k).__name__}",
            suggestion="Provide a positive integer for number of results",
            context={"received_type": type(top_k).__name__},
        )

    # Range checks
    if top_k <= 0:
        raise ValidationError(
            f"top_k must be positive, got {top_k}",
            suggestion="Use a positive integer (e.g., 5, 10, 20)",
            context={"top_k": top_k},
        )

    if top_k > max_top_k:
        raise ValidationError(
            f"top_k too large: {top_k} (maximum: {max_top_k})",
            suggestion=f"Use a top_k of {max_top_k} or less",
            context={"top_k": top_k, "max_top_k": max_top_k},
        )

    return top_k


def validate_memory_id(memory_id: str) -> str:
    """Validate memory ID.

    Args:
        memory_id: Memory ID to validate

    Returns:
        Validated memory ID

    Raises:
        ValidationError: If memory ID is invalid

    Example:
        >>> validate_memory_id("python-tips")
        'python-tips'
        >>> validate_memory_id("invalid id!")
        ValidationError: Invalid memory ID format
    """
    # Type check
    if not isinstance(memory_id, str):
        raise ValidationError(
            f"Memory ID must be a string, got {type(memory_id).__name__}",
            suggestion="Provide a valid memory ID as a string",
            context={"received_type": type(memory_id).__name__},
        )

    # Empty check
    if not memory_id.strip():
        raise ValidationError(
            "Memory ID cannot be empty",
            suggestion="Provide a non-empty memory ID",
            context={"memory_id": repr(memory_id)},
        )

    # Format check (alphanumeric, hyphens, underscores)
    if not re.match(r"^[\w\-]+$", memory_id):
        raise ValidationError(
            f"Invalid memory ID format: '{memory_id}'",
            suggestion="Use only letters, numbers, hyphens, and underscores (e.g., 'python-tips', 'ml_basics')",
            context={"memory_id": memory_id},
        )

    return memory_id


# ============================================================================
# Validation Decorators
# ============================================================================


def validate_inputs(**validators):
    """Decorator to validate function inputs.

    Args:
        **validators: Mapping of parameter names to validator functions

    Example:
        @validate_inputs(query=validate_query, depth=validate_depth)
        def search(query: str, depth: int = 2):
            ...
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect

            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            # Validate each parameter
            for param_name, validator in validators.items():
                if param_name in bound.arguments:
                    value = bound.arguments[param_name]
                    try:
                        validated = validator(value)
                        bound.arguments[param_name] = validated
                    except MemoGraphError:
                        raise
                    except Exception as e:
                        raise ValidationError(
                            f"Validation failed for parameter '{param_name}': {str(e)}",
                            context={"parameter": param_name, "value": value},
                        )

            # Call function with validated arguments
            return func(*bound.args, **bound.kwargs)

        return wrapper

    return decorator
