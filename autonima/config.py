"""Configuration management and validation for Autonima."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union

from .models.types import (
    PipelineConfig,
    SearchConfig,
    ScreeningConfig,
    RetrievalConfig,
    OutputConfig
)


class ConfigurationError(Exception):
    """Raised when there are issues with configuration loading or validation.
    """
    pass


class ConfigManager:
    """Manages loading and validation of Autonima pipeline configurations."""

    def __init__(self):
        self._config: Optional[PipelineConfig] = None

    def load_from_file(self, config_path: Union[str, Path]) -> PipelineConfig:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to the YAML configuration file

        Returns:
            PipelineConfig: Validated configuration object

        Raises:
            ConfigurationError: If configuration is invalid or file not found
        """
        config_path = Path(config_path)

        if not config_path.exists():
            msg = f"Configuration file not found: {config_path}"
            raise ConfigurationError(msg)

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            msg = f"Invalid YAML in configuration file: {e}"
            raise ConfigurationError(msg)
        except Exception as e:
            raise ConfigurationError(f"Error reading configuration file: {e}")

        if config_data is None:
            raise ConfigurationError("Configuration file is empty")

        return self.load_from_dict(config_data)

    def load_from_dict(self, config_dict: Dict[str, Any]) -> PipelineConfig:
        """
        Load configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            PipelineConfig: Validated configuration object

        Raises:
            ConfigurationError: If configuration is invalid
        """
        try:
            # Validate required fields
            if 'objective' not in config_dict:
                raise ConfigurationError("Missing required field: 'objective'")

            # Build nested configurations
            search_config = SearchConfig(**config_dict.get('search', {}))

            screening_config = ScreeningConfig()
            if 'screening' in config_dict:
                screening_dict = config_dict['screening']
                if 'abstract' in screening_dict:
                    screening_config.abstract.update(
                        screening_dict['abstract']
                    )
                if 'fulltext' in screening_dict:
                    screening_config.fulltext.update(
                        screening_dict['fulltext']
                    )

            retrieval_config = RetrievalConfig(
                **config_dict.get('retrieval', {})
            )
            output_config = OutputConfig(**config_dict.get('output', {}))

            # Create main config
            config = PipelineConfig(
                objective=config_dict['objective'],
                search=search_config,
                inclusion_criteria=config_dict.get('inclusion_criteria', []),
                exclusion_criteria=config_dict.get('exclusion_criteria', []),
                screening=screening_config,
                retrieval=retrieval_config,
                output=output_config
            )

            self._validate_config(config)
            self._config = config
            return config

        except Exception as e:
            raise ConfigurationError(f"Error parsing configuration: {e}")

    def _validate_config(self, config: PipelineConfig) -> None:
        """
        Validate the loaded configuration.

        Args:
            config: Configuration to validate

        Raises:
            ConfigurationError: If configuration is invalid
        """
        # Validate objective
        if not config.objective.strip():
            raise ConfigurationError("Objective cannot be empty")

        # Validate search configuration
        if not config.search.query.strip():
            raise ConfigurationError("Search query cannot be empty")

        if config.search.max_results <= 0:
            raise ConfigurationError("max_results must be positive")

        if config.search.database not in ['pubmed', 'pmc']:
            raise ConfigurationError("database must be 'pubmed' or 'pmc'")

        # Validate inclusion/exclusion criteria
        if not config.inclusion_criteria:
            raise ConfigurationError(
                "At least one inclusion criterion required"
            )
        
        # Validate screening thresholds and confidence reporting
        abstract_config = config.screening.abstract
        fulltext_config = config.screening.fulltext
        
        # Validate confidence reporting flags
        if not isinstance(
            abstract_config.get('confidence_reporting', False), bool
        ):
            raise ConfigurationError(
                "Abstract screening confidence_reporting must be a boolean"
            )
            
        if not isinstance(
            fulltext_config.get('confidence_reporting', False), bool
        ):
            raise ConfigurationError(
                "Full-text screening confidence_reporting must be a boolean"
            )
        
        # Validate screening thresholds (only if confidence reporting is enabled)
        if abstract_config.get('confidence_reporting', False):
            threshold = abstract_config.get('threshold', 0.75)
            if threshold is not None and not 0.0 <= threshold <= 1.0:
                raise ConfigurationError(
                    "Abstract screening threshold must be between 0.0 and 1.0"
                )
        
        if fulltext_config.get('confidence_reporting', False):
            threshold = fulltext_config.get('threshold', 0.8)
            if threshold is not None and not 0.0 <= threshold <= 1.0:
                raise ConfigurationError(
                    "Full-text screening threshold must be between 0.0 and 1.0"
                )
        
        # Validate output directory
        if not config.output.directory.strip():
            raise ConfigurationError("Output directory cannot be empty")

    def get_config(self) -> PipelineConfig:
        """Get the currently loaded configuration."""
        if self._config is None:
            raise ConfigurationError("No configuration loaded")
        return self._config

    def save_config(self, config_path: Union[str, Path], config: Optional[PipelineConfig] = None) -> None:
        """
        Save configuration to YAML file.

        Args:
            config_path: Path to save the configuration
            config: Configuration to save (uses loaded config if None)
        """
        if config is None:
            config = self.get_config()

        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = config.to_dict()

        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

    def create_sample_config(self) -> PipelineConfig:
        """Create a sample configuration for demonstration purposes."""
        return PipelineConfig(
            objective="Identify fMRI studies of working memory in schizophrenia",
            search=SearchConfig(
                database="pubmed",
                query="schizophrenia AND working memory AND fMRI",
                max_results=1000
            ),
            inclusion_criteria=[
                "Human participants",
                "fMRI neuroimaging",
                "Case-control or experimental design",
                "Task-based working memory paradigm"
            ],
            exclusion_criteria=[
                "Animal studies",
                "Review articles, meta-analyses, or theoretical papers",
                "Non-fMRI imaging modalities (e.g., PET, EEG, MEG)",
                "Studies without a control or comparison group"
            ],
            screening=ScreeningConfig(),
            retrieval=RetrievalConfig(),
            output=OutputConfig()
        )


def load_config(config_path: Union[str, Path]) -> PipelineConfig:
    """
    Convenience function to load a configuration from file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        PipelineConfig: Loaded and validated configuration
    """
    manager = ConfigManager()
    return manager.load_from_file(config_path)


def create_sample_config_file(output_path: Union[str, Path]) -> None:
    """
    Create a sample configuration file.

    Args:
        output_path: Path where to save the sample configuration
    """
    manager = ConfigManager()
    sample_config = manager.create_sample_config()
    manager.save_config(output_path, sample_config)