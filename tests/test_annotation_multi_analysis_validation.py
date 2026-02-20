"""Validation tests for strict multi-analysis annotation outputs."""

import pytest
from unittest.mock import patch, MagicMock

from autonima.annotation.client import AnnotationClient
from autonima.annotation.schema import (
    AnnotationCriteriaConfig,
    StudyAnalysisGroup,
    AnalysisMetadata,
    build_dynamic_multi_annotation_models,
)


def _criteria_list():
    return [
        AnnotationCriteriaConfig(
            name="affiliation_attachment",
            criteria_mapping={
                "inclusion": {
                    "GLOBAL_I1": "Healthy adults",
                    "AFFILIATION_ATTACHMENT_I1": "Affiliation contrast",
                },
                "exclusion": {
                    "GLOBAL_E1": "ROI-only analysis",
                },
            },
        ),
        AnnotationCriteriaConfig(
            name="social_communication",
            criteria_mapping={
                "inclusion": {
                    "GLOBAL_I1": "Healthy adults",
                    "SOCIAL_COMMUNICATION_I1": "Social communication contrast",
                },
                "exclusion": {
                    "GLOBAL_E1": "ROI-only analysis",
                },
            },
        ),
    ]


def test_dynamic_model_accepts_namespaced_ids():
    """Valid namespaced IDs should pass dynamic model validation."""
    _, output_model = build_dynamic_multi_annotation_models(_criteria_list())
    parsed = output_model(
        decisions=[
            {
                "annotation_name": "affiliation_attachment",
                "include": True,
                "reasoning": "Meets criteria",
                "inclusion_criteria_applied": [
                    "GLOBAL_I1",
                    "AFFILIATION_ATTACHMENT_I1",
                ],
                "exclusion_criteria_applied": [],
            },
            {
                "annotation_name": "social_communication",
                "include": False,
                "reasoning": "Missing inclusion criteria: GLOBAL_I1 SOCIAL_COMMUNICATION_I1",
                "inclusion_criteria_applied": [],
                "exclusion_criteria_applied": [],
            },
        ]
    )
    assert len(parsed.decisions) == 2


def test_dynamic_model_rejects_include_without_inclusion_ids():
    """include=true must provide non-empty inclusion_criteria_applied."""
    _, output_model = build_dynamic_multi_annotation_models(_criteria_list())
    with pytest.raises(ValueError, match="include=true"):
        output_model(
            decisions=[
                {
                    "annotation_name": "affiliation_attachment",
                    "include": True,
                    "reasoning": "Meets criteria",
                    "inclusion_criteria_applied": [],
                    "exclusion_criteria_applied": [],
                },
                {
                    "annotation_name": "social_communication",
                    "include": False,
                    "reasoning": "Missing inclusion criteria: GLOBAL_I1 SOCIAL_COMMUNICATION_I1",
                    "inclusion_criteria_applied": [],
                    "exclusion_criteria_applied": [],
                },
            ]
        )


def test_dynamic_model_rejects_exclude_without_exclusions_or_missing_ids_reason():
    """include=false without exclusions must mention missing inclusion IDs in reasoning."""
    _, output_model = build_dynamic_multi_annotation_models(_criteria_list())
    with pytest.raises(ValueError, match="missing/not-met inclusion criteria IDs"):
        output_model(
            decisions=[
                {
                    "annotation_name": "affiliation_attachment",
                    "include": True,
                    "reasoning": "Meets criteria",
                    "inclusion_criteria_applied": [
                        "GLOBAL_I1",
                        "AFFILIATION_ATTACHMENT_I1",
                    ],
                    "exclusion_criteria_applied": [],
                },
                {
                    "annotation_name": "social_communication",
                    "include": False,
                    "reasoning": "Not included for this construct",
                    "inclusion_criteria_applied": [],
                    "exclusion_criteria_applied": [],
                },
            ]
        )


def test_dynamic_model_rejects_invalid_criteria_membership():
    """IDs not in annotation mapping should be rejected."""
    _, output_model = build_dynamic_multi_annotation_models(_criteria_list())
    with pytest.raises(ValueError, match="Invalid inclusion_criteria_applied"):
        output_model(
            decisions=[
                {
                    "annotation_name": "affiliation_attachment",
                    "include": True,
                    "reasoning": "Meets criteria",
                    "inclusion_criteria_applied": [
                        "GLOBAL_I1",
                        "SOCIAL_COMMUNICATION_I1",
                    ],
                    "exclusion_criteria_applied": [],
                },
                {
                    "annotation_name": "social_communication",
                    "include": False,
                    "reasoning": "Missing inclusion criteria: GLOBAL_I1 SOCIAL_COMMUNICATION_I1",
                    "inclusion_criteria_applied": [],
                    "exclusion_criteria_applied": [],
                },
            ]
        )


def test_multi_analysis_client_uses_nested_schema_and_preserves_tags():
    """Client should validate nested multi-analysis output and preserve criteria tags."""
    criteria_list = _criteria_list()
    study_group = StudyAnalysisGroup(
        study_id="12345",
        analyses=[
            AnalysisMetadata(
                analysis_id="12345_analysis_0",
                study_id="12345",
                table_id="table1",
                analysis_name="Loved one > stranger",
            )
        ],
    )

    with patch("autonima.annotation.client.GenericLLMClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_function_call = MagicMock()
        mock_function_call.arguments = """
        {
          "study_id": "12345",
          "decisions": [
            {
              "analysis_id": "12345_analysis_0",
              "annotations": [
                {
                  "annotation_name": "affiliation_attachment",
                  "include": true,
                  "reasoning": "Meets global and local criteria",
                  "inclusion_criteria_applied": ["GLOBAL_I1", "AFFILIATION_ATTACHMENT_I1"],
                  "exclusion_criteria_applied": []
                },
                {
                  "annotation_name": "social_communication",
                  "include": false,
                  "reasoning": "Missing inclusion criteria: GLOBAL_I1 SOCIAL_COMMUNICATION_I1",
                  "inclusion_criteria_applied": [],
                  "exclusion_criteria_applied": []
                }
              ]
            }
          ]
        }
        """
        mock_message = MagicMock()
        mock_message.function_call = mock_function_call
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_api_response = MagicMock()
        mock_api_response.choices = [mock_choice]
        mock_client.client.chat.completions.create.return_value = mock_api_response

        client = AnnotationClient(max_retries=1)
        decisions = client.make_decision(
            metadata=study_group,
            criteria_list=criteria_list,
            metadata_fields=["analysis_name"],
            model="gpt-test",
            prompt_type="multi_analysis",
        )

        assert len(decisions) == 2
        assert decisions[0].inclusion_criteria_applied == [
            "GLOBAL_I1",
            "AFFILIATION_ATTACHMENT_I1",
        ]
        assert decisions[1].exclusion_criteria_applied == []

        call_kwargs = mock_client.client.chat.completions.create.call_args.kwargs
        function_schema = call_kwargs["functions"][0]["parameters"]
        assert "study_id" in function_schema.get("properties", {})
        assert "decisions" in function_schema.get("properties", {})

