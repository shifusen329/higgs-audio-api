"""
Text preprocessing module for TTS input normalization.

Provides YAML-configurable text preprocessing including:
- Abbreviation expansion (Dr. -> Doctor, e.g. -> for example)
- HTML entity handling (&amp; -> and)
- Unit conversions (°F -> degrees Fahrenheit)
- Whitespace normalization
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from loguru import logger

from .utils import full_to_half_width, remove_emoji


class TextPreprocessor:
    """
    YAML-configurable text preprocessor for TTS input.

    Applies regex-based transformations defined in a YAML config file.
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the text preprocessor.

        Args:
            config_path: Path to YAML config file. If None, uses default config.
        """
        if config_path is None:
            self.config_path = Path(__file__).parent / "config" / "pre_process_map.yaml"
        else:
            self.config_path = Path(config_path)

        self.rules: Dict[str, List[Dict[str, Any]]] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load preprocessing rules from YAML config."""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}. Using empty rules.")
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            if config is None:
                logger.warning("Empty config file. Using empty rules.")
                return

            # Compile regex patterns for each category
            for category, rules in config.items():
                if isinstance(rules, list):
                    compiled_rules = []
                    for rule in rules:
                        if "pattern" in rule and "replacement" in rule:
                            try:
                                compiled_rules.append({
                                    "pattern": re.compile(rule["pattern"]),
                                    "replacement": rule["replacement"],
                                })
                            except re.error as e:
                                logger.warning(f"Invalid regex pattern '{rule['pattern']}': {e}")
                    self.rules[category] = compiled_rules

            logger.info(f"Loaded {sum(len(r) for r in self.rules.values())} preprocessing rules from {self.config_path}")

        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML config: {e}")
        except Exception as e:
            logger.error(f"Failed to load config: {e}")

    def _apply_rules(self, text: str, categories: Optional[List[str]] = None) -> str:
        """
        Apply preprocessing rules to text.

        Args:
            text: Input text to preprocess.
            categories: List of rule categories to apply. If None, applies all.

        Returns:
            Preprocessed text.
        """
        if categories is None:
            categories = list(self.rules.keys())

        for category in categories:
            rules = self.rules.get(category, [])
            for rule in rules:
                text = rule["pattern"].sub(rule["replacement"], text)

        return text

    def preprocess(self, text: str) -> str:
        """
        Apply full preprocessing pipeline to text.

        Processing order:
        1. Remove emoji characters
        2. Convert full-width to half-width punctuation
        3. Apply abbreviation expansions
        4. Apply HTML entity conversions
        5. Apply unit conversions
        6. Apply symbol replacements
        7. Normalize whitespace

        Args:
            text: Input text to preprocess.

        Returns:
            Preprocessed text ready for TTS.
        """
        if not text or not text.strip():
            return ""

        # Step 1: Remove emojis
        text = remove_emoji(text)

        # Step 2: Convert full-width to half-width punctuation
        text = full_to_half_width(text)

        # Step 3-6: Apply YAML-configured rules in order
        rule_order = ["abbreviations", "html_entities", "units", "symbols"]
        for category in rule_order:
            if category in self.rules:
                text = self._apply_rules(text, [category])

        # Step 7: Normalize whitespace (applied last)
        if "whitespace" in self.rules:
            text = self._apply_rules(text, ["whitespace"])

        return text.strip()


# Global preprocessor instance for convenience
_default_preprocessor: Optional[TextPreprocessor] = None


def get_preprocessor(config_path: Optional[Union[str, Path]] = None) -> TextPreprocessor:
    """
    Get or create a text preprocessor instance.

    Args:
        config_path: Path to YAML config. If None and no instance exists,
                     creates one with default config.

    Returns:
        TextPreprocessor instance.
    """
    global _default_preprocessor

    if config_path is not None:
        return TextPreprocessor(config_path)

    if _default_preprocessor is None:
        _default_preprocessor = TextPreprocessor()

    return _default_preprocessor


def preprocess_text(text: str, config_path: Optional[Union[str, Path]] = None) -> str:
    """
    Convenience function to preprocess text.

    Args:
        text: Input text to preprocess.
        config_path: Optional path to YAML config.

    Returns:
        Preprocessed text.
    """
    preprocessor = get_preprocessor(config_path)
    return preprocessor.preprocess(text)
