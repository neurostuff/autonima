"""Configuration management and validation for Autonima."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union

from .models.types import (
    PipelineConfig,
    SearchConfig,
    ScreeningConfig,
    RetrievalConfig,
    ParsingConfig,
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

            retrieval_config = RetrievalConfig()
            if 'retrieval' in config_dict:
                retrieval_dict = config_dict['retrieval']
                # Handle backward compatibility for single full_text_source
                if 'full_text_source' in retrieval_dict:
                    if retrieval_dict['full_text_source'] is not None:
                        retrieval_config.full_text_sources = [retrieval_dict['full_text_source']]
                    # Remove the old key to avoid conflicts
                    retrieval_dict = {k: v for k, v in retrieval_dict.items() if k != 'full_text_source'}
                # Handle new full_text_sources
                if 'full_text_sources' in retrieval_dict:
                    retrieval_config.full_text_sources = retrieval_dict['full_text_sources']
                # Set other retrieval config values
                for key, value in retrieval_dict.items():
                    if hasattr(retrieval_config, key) and key != 'full_text_sources':
                        setattr(retrieval_config, key, value)
            output_config = OutputConfig(**config_dict.get('output', {}))

            # Create main config
            config = PipelineConfig(
                search=search_config,
                screening=screening_config,
                retrieval=retrieval_config,
                output=output_config,
                parsing=ParsingConfig(**config_dict.get('parsing', {})),
                annotation=self._load_annotation_config(config_dict.get('annotation', {}))
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
        # Validate search configuration
        if not config.search.query.strip():
            raise ConfigurationError("Search query cannot be empty")

        if config.search.max_results <= 0:
            raise ConfigurationError("max_results must be positive")

        if config.search.database not in ['pubmed', 'pmc']:
            raise ConfigurationError("database must be 'pubmed' or 'pmc'")

        # Validate screening configuration
        self._validate_screening_config(config)
        
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
            threshold = abstract_config.get('threshold', None)
            if threshold is not None and not 0.0 <= threshold <= 1.0:
                raise ConfigurationError(
                    "Abstract screening threshold must be between 0.0 and 1.0"
                )
        
        if fulltext_config.get('confidence_reporting', False):
            threshold = fulltext_config.get('threshold', None)
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
        # Create a screening config with example values
        screening_config = ScreeningConfig()
        # Example with confidence reporting enabled and threshold set
        screening_config.abstract.update({
            "model": "gpt-4o-mini",
            "confidence_reporting": True,
            "threshold": 0.9,
            "objective": "Identify fMRI studies of working memory in schizophrenia",
            "inclusion_criteria": [
                "Human participants",
                "fMRI neuroimaging",
                "Case-control or experimental design",
                "Task-based working memory paradigm"
            ],
            "exclusion_criteria": [
                "Animal studies",
                "Review articles, meta-analyses, or theoretical papers",
                "Non-fMRI imaging modalities (e.g., PET, EEG, MEG)",
                "Studies without a control or comparison group"
            ]
        })
        screening_config.fulltext.update({
            "model": "gpt-4",
            "confidence_reporting": True,
            "threshold": 0.95
        })
        
        return PipelineConfig(
            search=SearchConfig(
                database="pubmed",
                query="schizophrenia AND working memory AND fMRI",
                max_results=1000
            ),
            screening=screening_config,
            retrieval=RetrievalConfig(),
            output=OutputConfig(nimads=False),
            parsing=ParsingConfig(
                parse_coordinates=True,
                coordinate_model="gpt-4o-mini"
            ),
            annotation=self._load_annotation_config({
                "model": "gpt-4o-mini",
                "include_all_analyses": True,
                "annotations": []
            })
        )

    def _validate_screening_config(self, config: PipelineConfig) -> None:
        """
        Validate the screening configuration.

        Args:
            config: Configuration to validate

        Raises:
            ConfigurationError: If screening configuration is invalid
        """
        # Validate that at least one screening stage has an objective
        abstract_objective = config.screening.abstract.get('objective')
        fulltext_objective = config.screening.fulltext.get('objective')
        
        if not abstract_objective and not fulltext_objective:
            raise ConfigurationError(
                "At least one screening stage must have an objective"
            )
        
        # Validate that at least one screening stage has inclusion criteria
        abstract_inclusion = config.screening.abstract.get('inclusion_criteria')
        fulltext_inclusion = config.screening.fulltext.get('inclusion_criteria')
        
        if not abstract_inclusion and not fulltext_inclusion:
            raise ConfigurationError(
                "At least one screening stage must have inclusion criteria"
            )
        
        # Validate abstract screening if it has an objective
        if abstract_objective:
            if not abstract_inclusion:
                raise ConfigurationError(
                    "Abstract screening must have inclusion criteria when "
                    "objective is specified"
                )
        
        # Validate fulltext screening if it has an objective
        if fulltext_objective:
            if not fulltext_inclusion:
                raise ConfigurationError(
                    "Fulltext screening must have inclusion criteria when "
                    "objective is specified"
                )

    def _load_annotation_config(self, annotation_dict: Dict[str, Any]) -> 'AnnotationConfig':
        """
        Load annotation configuration from dictionary.
        
        Args:
            annotation_dict: Annotation configuration dictionary
            
        Returns:
            AnnotationConfig: Annotation configuration object
        """
        if not annotation_dict:
            from .annotation.schema import AnnotationConfig
            return AnnotationConfig()
        
        try:
            from .annotation.schema import AnnotationConfig, AnnotationCriteriaConfig
            
            # Extract annotations
            annotations = []
            if 'annotations' in annotation_dict:
                for criteria_dict in annotation_dict['annotations']:
                    criteria = AnnotationCriteriaConfig(**criteria_dict)
                    annotations.append(criteria)
            
            # Create annotation config
            annotation_config = AnnotationConfig(
                model=annotation_dict.get('model', 'gpt-4o-mini'),
                include_all_analyses=annotation_dict.get('include_all_analyses', True),
                annotations=annotations,
                enabled=annotation_dict.get('enabled', True),
                metadata_fields=annotation_dict.get('metadata_fields', [
                    "analysis_name",
                    "analysis_description",
                    "table_caption",
                    "study_title"
                ]),
                inclusion_criteria=annotation_dict.get('inclusion_criteria', []),
                exclusion_criteria=annotation_dict.get('exclusion_criteria', [])
            )
            
            return annotation_config
        except Exception as e:
            raise ConfigurationError(f"Error loading annotation configuration: {e}")


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