"""Utilities for managing inclusion/exclusion criteria with unique IDs."""

from typing import List, Dict, Optional
from dataclasses import dataclass, field
import json
import re
from pathlib import Path


@dataclass
class CriteriaMapping:
    """Mapping of criteria IDs to their text."""
    inclusion: Dict[str, str] = field(
        default_factory=dict)  # ID -> criterion text
    exclusion: Dict[str, str] = field(
        default_factory=dict)  # ID -> criterion text
    
    def to_dict(self) -> Dict[str, Dict[str, str]]:
        """Convert to dictionary for JSON serialization."""
        return {
            "inclusion": self.inclusion,
            "exclusion": self.exclusion
        }


class CriteriaIDAssigner:
    """Assigns unique IDs to inclusion/exclusion criteria globally."""
    
    def __init__(self):
        self.next_inclusion_id = 1
        self.next_exclusion_id = 1
    
    def assign_ids(
        self,
        inclusion_criteria: List[str],
        exclusion_criteria: List[str],
        inclusion_prefix: str = "I",
        exclusion_prefix: str = "E",
    ) -> CriteriaMapping:
        """
        Assign unique IDs to criteria lists.
        
        Args:
            inclusion_criteria: List of inclusion criterion strings
            exclusion_criteria: List of exclusion criterion strings
            
        Returns:
            CriteriaMapping with ID -> text mappings
        """
        inclusion_mapping = {}
        if inclusion_criteria:
            for criterion in inclusion_criteria:
                criterion_id = f"{inclusion_prefix}{self.next_inclusion_id}"
                inclusion_mapping[criterion_id] = criterion
                self.next_inclusion_id += 1
        
        exclusion_mapping = {}
        if exclusion_criteria:
            for criterion in exclusion_criteria:
                criterion_id = f"{exclusion_prefix}{self.next_exclusion_id}"
                exclusion_mapping[criterion_id] = criterion
                self.next_exclusion_id += 1
        
        return CriteriaMapping(
            inclusion=inclusion_mapping,
            exclusion=exclusion_mapping
        )


def sanitize_criteria_namespace(name: str) -> str:
    """Convert an annotation name into a stable uppercase criteria namespace."""
    namespace = re.sub(r"[^A-Za-z0-9]+", "_", name or "")
    namespace = re.sub(r"_+", "_", namespace).strip("_").upper()
    return namespace or "ANNOTATION"


def save_criteria_mapping(config, output_dir: str) -> None:
    """
    Save criteria ID mappings to JSON file.
    
    Args:
        config: Pipeline configuration containing criteria mappings
        output_dir: Directory to save the mapping file
    """
    output_path = Path(output_dir)
    mapping_file = output_path / "outputs" / "criteria_mapping.json"
    mapping_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Build mapping data structure
    mapping_data = {
        "screening": {
            "abstract": {
                "inclusion": {},
                "exclusion": {}
            },
            "fulltext": {
                "inclusion": {},
                "exclusion": {}
            }
        },
        "annotation": {
            "global": {
                "inclusion": {},
                "exclusion": {}
            },
            "annotations": {}
        }
    }
    
    # Add abstract screening criteria
    abstract_mapping = config.screening.abstract.get('criteria_mapping')
    if abstract_mapping:
        if isinstance(abstract_mapping, CriteriaMapping):
            mapping_dict = abstract_mapping.to_dict()
            mapping_data["screening"]["abstract"]["inclusion"] = (
                mapping_dict["inclusion"])
            mapping_data["screening"]["abstract"]["exclusion"] = (
                mapping_dict["exclusion"])
        else:
            mapping_data["screening"]["abstract"]["inclusion"] = (
                abstract_mapping.get("inclusion", {}))
            mapping_data["screening"]["abstract"]["exclusion"] = (
                abstract_mapping.get("exclusion", {}))
    
    # Add fulltext screening criteria
    fulltext_mapping = config.screening.fulltext.get('criteria_mapping')
    if fulltext_mapping:
        if isinstance(fulltext_mapping, CriteriaMapping):
            mapping_dict = fulltext_mapping.to_dict()
            mapping_data["screening"]["fulltext"]["inclusion"] = (
                mapping_dict["inclusion"])
            mapping_data["screening"]["fulltext"]["exclusion"] = (
                mapping_dict["exclusion"])
        else:
            mapping_data["screening"]["fulltext"]["inclusion"] = (
                fulltext_mapping.get("inclusion", {}))
            mapping_data["screening"]["fulltext"]["exclusion"] = (
                fulltext_mapping.get("exclusion", {}))
    
    # Add annotation criteria if present
    if hasattr(config, 'annotation'):
        # Global annotation criteria
        if (config.annotation.inclusion_criteria or
                config.annotation.exclusion_criteria):
            annotation_assigner = CriteriaIDAssigner()
            annotation_mapping = annotation_assigner.assign_ids(
                config.annotation.inclusion_criteria,
                config.annotation.exclusion_criteria,
                inclusion_prefix="GLOBAL_I",
                exclusion_prefix="GLOBAL_E",
            )
            mapping_dict = annotation_mapping.to_dict()
            mapping_data["annotation"]["global"]["inclusion"] = (
                mapping_dict["inclusion"])
            mapping_data["annotation"]["global"]["exclusion"] = (
                mapping_dict["exclusion"])
        
        # Per-annotation criteria
        for annotation in config.annotation.annotations:
            if annotation.inclusion_criteria or annotation.exclusion_criteria:
                namespace = sanitize_criteria_namespace(annotation.name)
                annotation_assigner = CriteriaIDAssigner()
                annotation_mapping = annotation_assigner.assign_ids(
                    annotation.inclusion_criteria,
                    annotation.exclusion_criteria,
                    inclusion_prefix=f"{namespace}_I",
                    exclusion_prefix=f"{namespace}_E",
                )
                mapping_dict = annotation_mapping.to_dict()
                mapping_data["annotation"]["annotations"][annotation.name] = {
                    "inclusion": mapping_dict["inclusion"],
                    "exclusion": mapping_dict["exclusion"]
                }
    
    # Save to file
    with open(mapping_file, 'w') as f:
        json.dump(mapping_data, f, indent=2)


def load_criteria_mapping(output_dir: str) -> Optional[Dict]:
    """
    Load criteria ID mappings from JSON file.
    
    Args:
        output_dir: Directory containing the mapping file
        
    Returns:
        Dictionary with criteria mappings or None if file doesn't exist
    """
    mapping_file = Path(output_dir) / "outputs" / "criteria_mapping.json"
    
    if not mapping_file.exists():
        return None
    
    try:
        with open(mapping_file, 'r') as f:
            return json.load(f)
    except Exception:
        return None
