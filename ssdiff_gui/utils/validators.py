"""Static file-path validation utilities for SSD.

Text, outcome, and lexicon validation live on Project (self-owned).
These path validators remain here because they don't need Project state
and are called before a project exists (e.g. in file-open dialogs).
"""

from pathlib import Path
from typing import List, Tuple


class Validator:
    """Static file-path validators."""

    @staticmethod
    def validate_embeddings_path(path: str) -> Tuple[List[str], List[str]]:
        """Validate an embeddings file path.

        Returns:
            Tuple of (errors, warnings) lists
        """
        errors: List[str] = []
        warnings: List[str] = []

        if not path:
            errors.append("No embedding file specified")
            return errors, warnings

        p = Path(path)
        if not p.exists():
            errors.append(f"File not found: {path}")
            return errors, warnings

        valid_extensions = {".kv", ".bin", ".txt", ".gz", ".vec"}
        if p.suffix.lower() not in valid_extensions:
            warnings.append(
                f"Unusual file extension: {p.suffix}. "
                f"Expected one of {valid_extensions}"
            )

        size_mb = p.stat().st_size / (1024 * 1024)
        if size_mb > 5000:
            warnings.append(
                f"Large embedding file ({size_mb:.0f} MB) - loading may take time"
            )

        return errors, warnings

    @staticmethod
    def validate_csv_path(path: str) -> Tuple[List[str], List[str]]:
        """Validate a CSV file path.

        Returns:
            Tuple of (errors, warnings) lists
        """
        errors: List[str] = []
        warnings: List[str] = []

        if not path:
            errors.append("No CSV file specified")
            return errors, warnings

        p = Path(path)
        if not p.exists():
            errors.append(f"File not found: {path}")
            return errors, warnings

        if p.suffix.lower() not in {".csv", ".tsv", ".txt"}:
            warnings.append(f"Unusual file extension: {p.suffix}")

        return errors, warnings
