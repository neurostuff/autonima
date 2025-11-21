import unittest
from autonima.annotation.schema import (
    TableMetadata, AnalysisMetadata, AnnotationCriteriaConfig
)
from autonima.annotation.prompts import create_table_grouped_prompt


class TestTableGroupedPrompt(unittest.TestCase):
    def test_basic_prompt_generation(self):
        # Create table metadata
        table_meta = TableMetadata(
            table_id="t123",
            caption="fMRI Results for Motor Task",
            footer="* p < 0.05, ** p < 0.01"
        )
        
        # Create analyses
        analyses = [
            AnalysisMetadata(
                analysis_id="a1",
                study_id="s123",
                table_id="t123",
                analysis_name="Main Analysis",
                analysis_description="Contrast: Motor vs Rest",
                study_title="Motor Cortex Activation Study"
            ),
            AnalysisMetadata(
                analysis_id="a2",
                study_id="s123",
                table_id="t123",
                analysis_name="Control Analysis",
                analysis_description="Contrast: Control Task",
                study_title="Motor Cortex Activation Study"
            )
        ]
        
        # Create criteria
        criteria = AnnotationCriteriaConfig(
            name="Motor Cortex",
            description="Activations in motor cortex regions",
            inclusion_criteria=[
                "Activates precentral gyrus",
                "Coordinates in M1"
            ],
            exclusion_criteria=["Activates visual cortex"]
        )
        
        # Generate prompt
        prompt = create_table_grouped_prompt(table_meta, analyses, criteria)
        
        # Verify key components
        self.assertIn("TABLE CONTEXT", prompt)
        self.assertIn("fMRI Results for Motor Task", prompt)
        self.assertIn("ANALYSES", prompt)
        self.assertIn("Analysis 1:", prompt)
        self.assertIn("Main Analysis", prompt)
        self.assertIn("Analysis 2:", prompt)
        self.assertIn("Control Analysis", prompt)
        self.assertIn("ANNOTATION CRITERIA", prompt)
        self.assertIn("Motor Cortex", prompt)
        self.assertIn("INCLUSION CRITERIA", prompt)
        self.assertIn("Activates precentral gyrus", prompt)
        self.assertIn("EXCLUSION CRITERIA", prompt)
        self.assertIn("Activates visual cortex", prompt)
        self.assertIn('"table_id": "t123"', prompt)
        self.assertIn('"decisions": [', prompt)
        self.assertIn('"analysis_id": "id1"', prompt)
        self.assertIn('"analysis_id": "id2"', prompt)

    def test_missing_metadata(self):
        # Create table metadata with missing fields
        table_meta = TableMetadata(table_id="t124")
        
        # Create minimal analysis
        analyses = [AnalysisMetadata(
            analysis_id="a3",
            study_id="s456",
            table_id="t124"
        )]
        
        # Create criteria
        criteria = AnnotationCriteriaConfig(name="Minimal Criteria")
        
        # Generate prompt
        prompt = create_table_grouped_prompt(table_meta, analyses, criteria)
        
        # Verify key components
        self.assertIn("TABLE CONTEXT", prompt)
        self.assertIn("No table context available", prompt)
        self.assertIn("ANALYSES", prompt)
        self.assertIn("Analysis 1:", prompt)
        self.assertIn("ANNOTATION CRITERIA", prompt)
        self.assertIn("Minimal Criteria", prompt)


if __name__ == "__main__":
    unittest.main()