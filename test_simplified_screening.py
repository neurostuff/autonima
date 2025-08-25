#!/usr/bin/env python3
"""Test the simplified screening module."""

import asyncio
import sys
from pathlib import Path

# Add the autonima package to the path
sys.path.insert(0, str(Path(__file__).parent))

from autonima.models.types import Study, StudyStatus, ScreeningConfig
from autonima.screening import LLMScreener


async def test_unified_screener():
    """Test the unified LLMScreener functionality."""
    print("Testing unified LLMScreener...")

    # Create a test study
    study = Study(
        pmid="TEST001",
        title="fMRI study of working memory in schizophrenia",
        abstract="This study investigated the neural correlates of working memory deficits in schizophrenia using fMRI. Participants included 50 patients with schizophrenia and 50 healthy controls. Results showed altered activation in prefrontal cortex during working memory tasks.",
        authors=["Smith J", "Johnson A"],
        journal="Neuroimage",
        publication_date="2023",
        status=StudyStatus.PENDING
    )

    # Create screening config with inclusion/exclusion criteria
    config = ScreeningConfig()
    
    # Add some test criteria
    config.inclusion_criteria = ["fMRI neuroimaging", "Human participants"]
    config.exclusion_criteria = ["Animal studies", "Review articles"]

    # Create unified screener
    screener = LLMScreener(config)

    # Test abstract screening
    print("Testing abstract screening...")
    abstract_results = await screener.screen_abstracts([study])
    
    print(f"Abstract screening results: {len(abstract_results)}")
    for result in abstract_results:
        print(f"  Study: {result.study_id}")
        print(f"  Decision: {result.decision}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Reason: {result.reason}")

    # Test full-text screening (with a study that has full text path)
    study_with_fulltext = Study(
        pmid="TEST002",
        title="Comprehensive fMRI analysis",
        abstract="Detailed study...",
        authors=["Williams K"],
        journal="Brain Research",
        publication_date="2023",
        status=StudyStatus.INCLUDED,
        full_text_path="downloads/TEST002.pdf"
    )

    print("\nTesting full-text screening...")
    fulltext_results = await screener.screen_fulltexts([study_with_fulltext])
    
    print(f"Full-text screening results: {len(fulltext_results)}")
    for result in fulltext_results:
        print(f"  Study: {result.study_id}")
        print(f"  Decision: {result.decision}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Reason: {result.reason}")

    # Test screening info
    info = screener.get_screening_info()
    print(f"\nScreening info: {info}")

    print("\n✅ Unified LLMScreener tests completed!")
    return abstract_results, fulltext_results


async def main():
    """Run all tests."""
    print("Running simplified screening module tests...\n")

    try:
        # Test unified screener
        abstract_results, fulltext_results = await test_unified_screener()

        print(f"\n✅ All tests completed!")
        print(f"Abstract screening: {len(abstract_results)} results")
        print(f"Full-text screening: {len(fulltext_results)} results")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)