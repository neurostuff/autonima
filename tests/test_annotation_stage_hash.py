"""The annotation stage hash must not move when a new config field sits at its default."""

from autonima.annotation.processor import AnnotationProcessor
from autonima.annotation.schema import AnnotationConfig, AnnotationCriteriaConfig


def _config(names=("A", "B")):
    return AnnotationConfig(
        create_all_included_annotations=False,
        metadata_fields=["analysis_name"],
        annotations=[
            AnnotationCriteriaConfig(name=name, inclusion_criteria=[f"{name} criterion"])
            for name in names
        ],
    )


def test_new_config_fields_at_their_default_leave_the_stage_hash_alone():
    """Regression: `backend` and `additional_instructions` invalidated every earlier cache."""
    from autonima.annotation.prompts import ANNOTATION_PROMPT_VERSION
    from autonima.backends.jev import POST_HOC_CONFIG_KEYS
    from autonima.execution import stable_hash

    config = _config()
    before_fields = {
        k: v for k, v in config.model_dump().items()
        if k not in POST_HOC_CONFIG_KEYS | {"model_params", "backend", "additional_instructions"}
    }
    before_fields["annotations"] = [
        {k: v for k, v in a.items() if k != "additional_instructions"}
        for a in before_fields["annotations"]
    ]
    pre_field_hash = stable_hash({**before_fields, "prompt_version": ANNOTATION_PROMPT_VERSION})
    assert AnnotationProcessor(config).stage_hash == pre_field_hash

    # Once actually used, they change what is asked, so they must change the hash.
    guided = _config()
    guided.additional_instructions = "Treat analysis_0 labels as generic."
    assert AnnotationProcessor(guided).stage_hash != pre_field_hash
    per_target = _config()
    per_target.annotations[0].additional_instructions = "Only whole-brain maps."
    assert AnnotationProcessor(per_target).stage_hash != pre_field_hash
