"""Tests for input validation module.

This test suite validates:
- Query validation
- Path validation and security
- Tag validation and normalization
- Parameter validation (salience, depth, top_k)
- Custom exception classes
- Error messages with suggestions
"""

import pytest

from memograph.core.validation import (
    MemoGraphError,
    SecurityError,
    ValidationError,
    validate_depth,
    validate_memory_id,
    validate_path,
    validate_query,
    validate_salience,
    validate_tags,
    validate_top_k,
)


class TestQueryValidation:
    """Test query string validation."""

    def test_valid_query(self):
        """Test valid query strings."""
        assert validate_query("python tips") == "python tips"
        assert validate_query("  machine learning  ") == "machine learning"
        assert validate_query("a" * 100) == "a" * 100

    def test_empty_query(self):
        """Test empty query rejection."""
        with pytest.raises(ValidationError) as exc_info:
            validate_query("")

        assert "cannot be empty" in str(exc_info.value)
        assert "suggestion" in str(exc_info.value).lower()

    def test_whitespace_query(self):
        """Test whitespace-only query rejection."""
        with pytest.raises(ValidationError):
            validate_query("   ")

    def test_query_too_short(self):
        """Test minimum length validation."""
        with pytest.raises(ValidationError) as exc_info:
            validate_query("a", min_length=5)

        assert "too short" in str(exc_info.value).lower()

    def test_query_too_long(self):
        """Test maximum length validation."""
        with pytest.raises(ValidationError) as exc_info:
            validate_query("a" * 1001, max_length=1000)

        assert "too long" in str(exc_info.value).lower()

    def test_invalid_type(self):
        """Test type validation."""
        with pytest.raises(ValidationError) as exc_info:
            validate_query(123)

        assert "must be a string" in str(exc_info.value)


class TestPathValidation:
    """Test path validation and security."""

    def test_valid_path(self, tmp_path):
        """Test valid path."""
        test_file = tmp_path / "test.md"
        test_file.touch()

        result = validate_path(test_file, must_exist=True, must_be_file=True)
        assert result == test_file

    def test_path_must_exist(self, tmp_path):
        """Test existence check."""
        non_existent = tmp_path / "nonexistent.md"

        with pytest.raises(ValidationError) as exc_info:
            validate_path(non_existent, must_exist=True)

        assert "does not exist" in str(exc_info.value)

    def test_path_must_be_file(self, tmp_path):
        """Test file type check."""
        with pytest.raises(ValidationError) as exc_info:
            validate_path(tmp_path, must_exist=True, must_be_file=True)

        assert "not a file" in str(exc_info.value)

    def test_path_must_be_dir(self, tmp_path):
        """Test directory type check."""
        test_file = tmp_path / "test.md"
        test_file.touch()

        with pytest.raises(ValidationError) as exc_info:
            validate_path(test_file, must_exist=True, must_be_dir=True)

        assert "not a directory" in str(exc_info.value)

    def test_allowed_extensions(self, tmp_path):
        """Test extension validation."""
        test_file = tmp_path / "test.txt"
        test_file.touch()

        with pytest.raises(ValidationError) as exc_info:
            validate_path(
                test_file,
                must_exist=True,
                allowed_extensions=['.md', '.markdown']
            )

        assert "invalid file extension" in str(exc_info.value).lower()

    def test_path_traversal_security(self, tmp_path):
        """Test path traversal prevention."""
        base_path = tmp_path / "vault"
        base_path.mkdir()

        # Try to access parent directory
        malicious_path = base_path / ".." / ".." / "etc" / "passwd"

        with pytest.raises(SecurityError) as exc_info:
            validate_path(malicious_path, base_path=base_path)

        assert "path traversal" in str(exc_info.value).lower()

    def test_invalid_type(self):
        """Test type validation."""
        with pytest.raises(ValidationError):
            validate_path(123)


class TestTagValidation:
    """Test tag validation and normalization."""

    def test_single_tag(self):
        """Test single tag string."""
        assert validate_tags("python") == ["python"]

    def test_tag_list(self):
        """Test list of tags."""
        result = validate_tags(["python", "machine-learning", "AI"])
        assert result == ["python", "machine-learning", "ai"]

    def test_tag_normalization(self):
        """Test tag normalization."""
        result = validate_tags(["  Python  ", "Machine Learning", "AI!!!"])
        assert result == ["python", "machinelearning", "ai"]

    def test_duplicate_removal(self):
        """Test duplicate tag removal."""
        result = validate_tags(["python", "Python", "PYTHON"])
        assert result == ["python"]

    def test_empty_tag_removal(self):
        """Test empty tag removal."""
        result = validate_tags(["python", "", "  ", "ml"])
        assert result == ["python", "ml"]

    def test_too_many_tags(self):
        """Test maximum tag count."""
        tags = [f"tag{i}" for i in range(25)]

        with pytest.raises(ValidationError) as exc_info:
            validate_tags(tags, max_tags=20)

        assert "too many tags" in str(exc_info.value).lower()

    def test_tag_too_long(self):
        """Test maximum tag length."""
        with pytest.raises(ValidationError) as exc_info:
            validate_tags("a" * 100, max_tag_length=50)

        assert "too long" in str(exc_info.value).lower()

    def test_invalid_type(self):
        """Test type validation."""
        with pytest.raises(ValidationError):
            validate_tags(123)


class TestSalienceValidation:
    """Test salience score validation."""

    def test_valid_salience(self):
        """Test valid salience values."""
        assert validate_salience(0.0) == 0.0
        assert validate_salience(0.5) == 0.5
        assert validate_salience(1.0) == 1.0

    def test_salience_out_of_range(self):
        """Test out of range rejection."""
        with pytest.raises(ValidationError) as exc_info:
            validate_salience(1.5)

        assert "between 0.0 and 1.0" in str(exc_info.value)

        with pytest.raises(ValidationError):
            validate_salience(-0.1)

    def test_invalid_type(self):
        """Test type validation."""
        with pytest.raises(ValidationError):
            validate_salience("high")


class TestDepthValidation:
    """Test graph traversal depth validation."""

    def test_valid_depth(self):
        """Test valid depth values."""
        assert validate_depth(0) == 0
        assert validate_depth(2) == 2
        assert validate_depth(10) == 10

    def test_negative_depth(self):
        """Test negative depth rejection."""
        with pytest.raises(ValidationError) as exc_info:
            validate_depth(-1)

        assert "must be non-negative" in str(exc_info.value)

    def test_depth_too_large(self):
        """Test maximum depth."""
        with pytest.raises(ValidationError) as exc_info:
            validate_depth(20, max_depth=10)

        assert "too large" in str(exc_info.value).lower()

    def test_invalid_type(self):
        """Test type validation."""
        with pytest.raises(ValidationError):
            validate_depth(2.5)


class TestTopKValidation:
    """Test top-k parameter validation."""

    def test_valid_top_k(self):
        """Test valid top-k values."""
        assert validate_top_k(1) == 1
        assert validate_top_k(10) == 10
        assert validate_top_k(100) == 100

    def test_zero_top_k(self):
        """Test zero rejection."""
        with pytest.raises(ValidationError) as exc_info:
            validate_top_k(0)

        assert "must be positive" in str(exc_info.value)

    def test_negative_top_k(self):
        """Test negative rejection."""
        with pytest.raises(ValidationError):
            validate_top_k(-5)

    def test_top_k_too_large(self):
        """Test maximum top-k."""
        with pytest.raises(ValidationError) as exc_info:
            validate_top_k(200, max_top_k=100)

        assert "too large" in str(exc_info.value).lower()

    def test_invalid_type(self):
        """Test type validation."""
        with pytest.raises(ValidationError):
            validate_top_k(10.5)


class TestMemoryIdValidation:
    """Test memory ID validation."""

    def test_valid_memory_id(self):
        """Test valid memory IDs."""
        assert validate_memory_id("python-tips") == "python-tips"
        assert validate_memory_id("ml_basics") == "ml_basics"
        assert validate_memory_id("note123") == "note123"

    def test_empty_memory_id(self):
        """Test empty ID rejection."""
        with pytest.raises(ValidationError):
            validate_memory_id("")

    def test_invalid_characters(self):
        """Test invalid character rejection."""
        with pytest.raises(ValidationError) as exc_info:
            validate_memory_id("invalid id!")

        assert "invalid" in str(exc_info.value).lower()
        assert "format" in str(exc_info.value).lower()

    def test_invalid_type(self):
        """Test type validation."""
        with pytest.raises(ValidationError):
            validate_memory_id(123)


class TestCustomExceptions:
    """Test custom exception classes."""

    def test_memograph_error(self):
        """Test base MemoGraphError."""
        error = MemoGraphError(
            "Something went wrong",
            suggestion="Try this fix",
            context={'key': 'value'}
        )

        error_str = str(error)
        assert "Something went wrong" in error_str
        assert "Try this fix" in error_str
        assert "key: value" in error_str

    def test_validation_error(self):
        """Test ValidationError."""
        error = ValidationError(
            "Invalid input",
            suggestion="Use valid input"
        )

        assert isinstance(error, MemoGraphError)
        assert "Invalid input" in str(error)

    def test_security_error(self):
        """Test SecurityError."""
        error = SecurityError(
            "Security violation",
            suggestion="Check permissions"
        )

        assert isinstance(error, MemoGraphError)
        assert "Security violation" in str(error)
