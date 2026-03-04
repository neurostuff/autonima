"""Configuration management and validation for Autonima."""

from importlib import resources
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
import yaml

from .models.types import (
    PipelineConfig,
    SearchConfig,
    ScreeningConfig,
    RetrievalConfig,
    ParsingConfig,
    OutputConfig
)
from .utils.criteria import CriteriaIDAssigner, save_criteria_mapping

logger = logging.getLogger(__name__)

_SAMPLE_CONFIG_PACKAGE = "autonima.templates"
_SAMPLE_CONFIG_NAME = "sample_config.yml"


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

            # Assign IDs to criteria
            id_assigner = CriteriaIDAssigner()
            
            # Assign IDs to abstract screening criteria
            abstract_inclusion = config.screening.abstract.get('inclusion_criteria', [])
            abstract_exclusion = config.screening.abstract.get('exclusion_criteria', [])
            abstract_mapping = id_assigner.assign_ids(abstract_inclusion, abstract_exclusion)
            config.screening.abstract['criteria_mapping'] = abstract_mapping
            
            # Assign IDs to fulltext screening criteria
            fulltext_inclusion = config.screening.fulltext.get('inclusion_criteria', [])
            fulltext_exclusion = config.screening.fulltext.get('exclusion_criteria', [])
            fulltext_mapping = id_assigner.assign_ids(fulltext_inclusion, fulltext_exclusion)
            config.screening.fulltext['criteria_mapping'] = fulltext_mapping

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
        if not (config.search.pmids_file or config.search.pmids_list):
            # Query-based search
            if not config.search.query.strip():
                raise ConfigurationError("Search query cannot be empty when not using PMIDs list/file")

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
        
        # config.retrieval.coordinates_path_templates is mutually exclusive with processed_data_path for each source
        for source in config.retrieval.full_text_sources:
            if hasattr(source, 'coordinates_path_templates') and hasattr(source, 'processed_data_path'):
                if source.coordinates_path_templates and source.processed_data_path:
                    raise ConfigurationError(
                        "coordinates_path_templates and processed_data_path are mutually exclusive for each source"
                    )

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
        """Create the canonical sample configuration as a validated object."""
        sample_config_text = get_sample_config_text()
        config_data = yaml.safe_load(sample_config_text)
        if config_data is None:
            raise ConfigurationError("Canonical sample configuration is empty")
        return self.load_from_dict(config_data)

    def _validate_screening_config(self, config: PipelineConfig) -> None:
        """
        Validate the screening configuration.

        Args:
            config: Configuration to validate

        Raises:
            ConfigurationError: If screening configuration is invalid
        """
        # Get screening configuration
        abstract_objective = config.screening.abstract.get('objective')
        fulltext_objective = config.screening.fulltext.get('objective')
        abstract_inclusion = config.screening.abstract.get('inclusion_criteria')
        fulltext_inclusion = config.screening.fulltext.get('inclusion_criteria')
        abstract_skip = config.screening.abstract.get('skip_stage', False)
        fulltext_skip = config.screening.fulltext.get('skip_stage', False)
        
        # Validate abstract screening
        if abstract_skip:
            # If skip_stage is True, objective should not be set
            if abstract_objective:
                logger.warning(
                    "Abstract screening has skip_stage=True but also has an objective. "
                    "The stage will be skipped and the objective will be ignored."
                )
        else:
            # If skip_stage is False (or not set), objective is required
            if not abstract_objective:
                raise ConfigurationError(
                    "Abstract screening must have an 'objective' or set 'skip_stage: true'"
                )
            # If objective is set, inclusion criteria are required
            if not abstract_inclusion:
                raise ConfigurationError(
                    "Abstract screening must have inclusion criteria when "
                    "objective is specified"
                )
        
        # Validate fulltext screening
        if fulltext_skip:
            # If skip_stage is True, objective should not be set
            if fulltext_objective:
                logger.warning(
                    "Fulltext screening has skip_stage=True but also has an objective. "
                    "The stage will be skipped and the objective will be ignored."
                )
        else:
            # If skip_stage is False (or not set), objective is required
            if not fulltext_objective:
                raise ConfigurationError(
                    "Fulltext screening must have an 'objective' or set 'skip_stage: true'"
                )
            # If objective is set, inclusion criteria are required
            if not fulltext_inclusion:
                raise ConfigurationError(
                    "Fulltext screening must have inclusion criteria when "
                    "objective is specified"
                )
        
        # Log a warning if both screening stages are skipped
        if abstract_skip and fulltext_skip:
            logger.warning(
                "Both abstract and fulltext screening stages are skipped. "
                "All studies from the search stage will pass to the next pipeline stage."
            )

    def _load_annotation_config(
        self, annotation_dict: Dict[str, Any]
    ) -> 'AnnotationConfig':
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
            from .annotation.schema import (
                AnnotationConfig,
                AnnotationCriteriaConfig
            )
            
            # Extract annotations
            annotations = []
            if 'annotations' in annotation_dict:
                for criteria_dict in annotation_dict['annotations']:
                    criteria = AnnotationCriteriaConfig(**criteria_dict)
                    annotations.append(criteria)
            
            # Get annotation configuration values
            create_all_included = annotation_dict.get(
                'create_all_included_annotation', True
            )
            
            create_all_from_search = annotation_dict.get(
                'create_all_from_search_annotation', False
            )
            
            # Create annotation config
            annotation_config = AnnotationConfig(
                model=annotation_dict.get('model', 'gpt-4o-mini'),
                create_all_included_annotation=create_all_included,
                create_all_from_search_annotation=create_all_from_search,
                annotations=annotations,
                enabled=annotation_dict.get('enabled', True),
                prompt_type=annotation_dict.get('prompt_type', 'single_analysis'),
                metadata_fields=annotation_dict.get('metadata_fields', [
                    "analysis_name",
                    "analysis_description",
                    "table_caption",
                    "study_title"
                ]),
                inclusion_criteria=annotation_dict.get(
                    'inclusion_criteria', []
                ),
                exclusion_criteria=annotation_dict.get(
                    'exclusion_criteria', []
                )
            )
            
            return annotation_config
        except Exception as e:
            raise ConfigurationError(
                f"Error loading annotation configuration: {e}"
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
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(get_sample_config_text(), encoding="utf-8")


def get_sample_config_text() -> str:
    """Return the canonical sample configuration YAML text."""
    if hasattr(resources, "files"):
        return (
            resources.files(_SAMPLE_CONFIG_PACKAGE)
            .joinpath(_SAMPLE_CONFIG_NAME)
            .read_text(encoding="utf-8")
        )
    return resources.read_text(
        _SAMPLE_CONFIG_PACKAGE,
        _SAMPLE_CONFIG_NAME,
        encoding="utf-8",
    )
