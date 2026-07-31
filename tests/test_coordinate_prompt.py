from autonima.coordinates.prompts import (
    COORDINATE_PARSING_PROMPT_VERSION,
    create_coordinate_parsing_prompt,
)
from autonima.execution import stage_signature_payloads


def test_coordinate_prompt_encodes_reviewed_boundary_rules():
    prompt = create_coordinate_parsing_prompt(
        "Contrast,Region,X,Y,Z\nA > B,Insula,1,2,3",
        table_caption="Activations and deactivations",
    )

    assert "activation versus deactivation" in prompt
    assert "columns named Contrast" in prompt
    assert "Brain region, lobe, cluster, hemisphere/left/right" in prompt
    assert "Extract each valid coordinate row exactly once" in prompt
    assert "Repeated blank cells inherit" in prompt
    assert "repeated statistic/X/Y/Z column groups" in prompt
    assert "left and right hemisphere sections alone never define" in prompt
    assert '"ROI analysis" and "whole-brain' in prompt
    assert "same nonnumeric label is repeated across most or all columns" in prompt
    assert 'When "all groups" and named subgroups' in prompt
    assert '"Predicted" and "not predicted"' in prompt
    assert "do not retroactively assign preceding" in prompt
    assert "route points by sign" in prompt
    assert "Never concatenate multiple tokens" in prompt
    assert "above 200 mm" in prompt


def test_parsing_stage_signature_tracks_prompt_version():
    payloads = stage_signature_payloads(
        {
            "parsing": {
                "parse_coordinates": True,
                "coordinate_model": "gpt-5-mini",
            }
        }
    )

    assert payloads["parsing"]["prompt_version"] == COORDINATE_PARSING_PROMPT_VERSION
