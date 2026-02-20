"""Pytest tests for criteria mapping functionality."""

import json
from types import SimpleNamespace

from autonima.utils.criteria import (
    CriteriaIDAssigner,
    CriteriaMapping,
    save_criteria_mapping,
)
from autonima.config import ConfigManager


def test_criteria_id_assigner():
    """Test CriteriaIDAssigner functionality."""
    # Test with inclusion criteria only
    assigner = CriteriaIDAssigner()
    inclusion_criteria = ["Human participants", "fMRI neuroimaging"]
    exclusion_criteria = []
    
    mapping = assigner.assign_ids(inclusion_criteria, exclusion_criteria)
    
    assert isinstance(mapping, CriteriaMapping)
    assert len(mapping.inclusion) == 2
    assert len(mapping.exclusion) == 0
    assert mapping.inclusion["I1"] == "Human participants"
    assert mapping.inclusion["I2"] == "fMRI neuroimaging"
    
    # Test with exclusion criteria only
    assigner = CriteriaIDAssigner()  # Reset counters
    inclusion_criteria = []
    exclusion_criteria = ["Animal studies", "Review articles"]
    
    mapping = assigner.assign_ids(inclusion_criteria, exclusion_criteria)
    
    assert isinstance(mapping, CriteriaMapping)
    assert len(mapping.inclusion) == 0
    assert len(mapping.exclusion) == 2
    assert mapping.exclusion["E1"] == "Animal studies"
    assert mapping.exclusion["E2"] == "Review articles"
    
    # Test with both inclusion and exclusion criteria
    assigner = CriteriaIDAssigner()  # Reset counters
    inclusion_criteria = ["Human participants", "fMRI neuroimaging"]
    exclusion_criteria = ["Animal studies", "Review articles"]
    
    mapping = assigner.assign_ids(inclusion_criteria, exclusion_criteria)
    
    assert isinstance(mapping, CriteriaMapping)
    assert len(mapping.inclusion) == 2
    assert len(mapping.exclusion) == 2
    assert mapping.inclusion["I1"] == "Human participants"
    assert mapping.inclusion["I2"] == "fMRI neuroimaging"
    assert mapping.exclusion["E1"] == "Animal studies"
    assert mapping.exclusion["E2"] == "Review articles"


def test_criteria_mapping_in_config():
    """Test that criteria mapping is added to configuration."""
    # Load the configuration through ConfigManager to trigger criteria mapping
    config_manager = ConfigManager()
    
    # Convert to dict and back to trigger the mapping
    config_dict = {
        "search": {
            "database": "pubmed",
            "query": "test query",
            "max_results": 10
        },
        "screening": {
            "abstract": {
                "objective": "Test objective",
                "inclusion_criteria": [
                    "Human participants",
                    "fMRI neuroimaging",
                    "Case-control or experimental design"
                ],
                "exclusion_criteria": [
                    "Animal studies",
                    "Review articles",
                    "Non-fMRI imaging"
                ]
                },
                "fulltext": {
                    "objective": "Fulltext test objective",
                    "inclusion_criteria": [
                        "Sample size > 10",
                        "Statistical significance reported"
                    ],
                "exclusion_criteria": [
                    "Insufficient methodological detail"
                ]
            }
        },
        "retrieval": {},
        "output": {
            "directory": "results"
        },
        "parsing": {},
        "annotation": {
            "annotations": [],
            "inclusion_criteria": [],
            "exclusion_criteria": []
        }
    }
    
    loaded_config = config_manager.load_from_dict(config_dict)
    
    # Check that criteria mapping was added to abstract screening
    assert 'criteria_mapping' in loaded_config.screening.abstract
    abstract_mapping = loaded_config.screening.abstract['criteria_mapping']
    assert len(abstract_mapping.inclusion) == 3
    assert len(abstract_mapping.exclusion) == 3
    assert abstract_mapping.inclusion["I1"] == "Human participants"
    assert abstract_mapping.inclusion["I2"] == "fMRI neuroimaging"
    assert (abstract_mapping.inclusion["I3"] == 
            "Case-control or experimental design")
    assert abstract_mapping.exclusion["E1"] == "Animal studies"
    assert abstract_mapping.exclusion["E2"] == "Review articles"
    assert abstract_mapping.exclusion["E3"] == "Non-fMRI imaging"
    
    # Check that criteria mapping was added to fulltext screening
    assert 'criteria_mapping' in loaded_config.screening.fulltext
    fulltext_mapping = loaded_config.screening.fulltext['criteria_mapping']
    assert len(fulltext_mapping.inclusion) == 2
    assert len(fulltext_mapping.exclusion) == 1
    assert fulltext_mapping.inclusion["I4"] == "Sample size > 10"
    assert (fulltext_mapping.inclusion["I5"] == 
            "Statistical significance reported")
    assert (fulltext_mapping.exclusion["E4"] == 
            "Insufficient methodological detail")


def test_criteria_mapping_with_none_values():
    """Test that criteria mapping handles None values correctly."""
    assigner = CriteriaIDAssigner()
    
    # Test with None values
    mapping = assigner.assign_ids(None, None)
    
    assert isinstance(mapping, CriteriaMapping)
    assert len(mapping.inclusion) == 0
    assert len(mapping.exclusion) == 0
    
    # Test with None inclusion criteria
    assigner = CriteriaIDAssigner()  # Reset counters
    exclusion_criteria = ["Animal studies", "Review articles"]
    mapping = assigner.assign_ids(None, exclusion_criteria)
    
    assert isinstance(mapping, CriteriaMapping)
    assert len(mapping.inclusion) == 0
    assert len(mapping.exclusion) == 2
    assert mapping.exclusion["E1"] == "Animal studies"
    assert mapping.exclusion["E2"] == "Review articles"
    
    # Test with None exclusion criteria
    assigner = CriteriaIDAssigner()  # Reset counters
    inclusion_criteria = ["Human participants", "fMRI neuroimaging"]
    mapping = assigner.assign_ids(inclusion_criteria, None)
    
    assert isinstance(mapping, CriteriaMapping)
    assert len(mapping.inclusion) == 2
    assert len(mapping.exclusion) == 0
    assert mapping.inclusion["I1"] == "Human participants"
    assert mapping.inclusion["I2"] == "fMRI neuroimaging"


def test_annotation_criteria_mapping_uses_namespaced_ids(tmp_path):
    """Annotation criteria IDs should be namespaced in saved mapping."""
    config = SimpleNamespace(
        screening=SimpleNamespace(
            abstract={"criteria_mapping": None},
            fulltext={"criteria_mapping": None},
        ),
        annotation=SimpleNamespace(
            inclusion_criteria=["Healthy adults"],
            exclusion_criteria=["ROI analyses"],
            annotations=[
                SimpleNamespace(
                    name="affiliation_attachment",
                    inclusion_criteria=["Affiliation contrast"],
                    exclusion_criteria=[],
                ),
                SimpleNamespace(
                    name="social_communication",
                    inclusion_criteria=["Communication contrast"],
                    exclusion_criteria=["Resting state"],
                ),
            ],
        ),
    )

    save_criteria_mapping(config, str(tmp_path))
    mapping_path = tmp_path / "outputs" / "criteria_mapping.json"
    saved = json.loads(mapping_path.read_text())
    ann = saved["annotation"]

    assert list(ann["global"]["inclusion"].keys()) == ["GLOBAL_I1"]
    assert list(ann["global"]["exclusion"].keys()) == ["GLOBAL_E1"]
    assert list(ann["annotations"]["affiliation_attachment"]["inclusion"].keys()) == [
        "AFFILIATION_ATTACHMENT_I1"
    ]
    assert list(ann["annotations"]["social_communication"]["inclusion"].keys()) == [
        "SOCIAL_COMMUNICATION_I1"
    ]
    assert list(ann["annotations"]["social_communication"]["exclusion"].keys()) == [
        "SOCIAL_COMMUNICATION_E1"
    ]
