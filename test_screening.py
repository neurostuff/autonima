#!/usr/bin/env python3
"""Basic tests for the screening module."""

import asyncio
import sys
from pathlib import Path

# Add the autonima package to the path
sys.path.insert(0, str(Path(__file__).parent))

from autonima.models.types import Study, StudyStatus, ScreeningConfig
from autonima.screening import AbstractScreener, FullTextScreener


async def test_abstract_screening():
    """Test abstract screening functionality."""
    print("Testing abstract screening...")

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

    # Create screening config
    config = ScreeningConfig()

    # Create abstract screener
    screener = AbstractScreener(config)

    # Test screening
    results = await screener.screen_abstracts([study])

    print(f"Screening results: {len(results)}")
    for result in results:
        print(f"  Study: {result.study_id}")
        print(f"  Decision: {result.decision}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Reason: {result.reason}")

    return results


async def test_fulltext_screening():
    """Test full-text screening functionality."""
    print("\nTesting full-text screening...")

    # Create a test study with full text path
    study = Study(
        pmid="TEST002",
        title="Comprehensive fMRI analysis of cognitive deficits in schizophrenia",
        abstract="Detailed study of cognitive impairments...",
        authors=["Williams K", "Brown L"],
        journal="Brain Research",
        publication_date="2023",
        status=StudyStatus.INCLUDED,
        full_text_path="downloads/TEST002.pdf"
    )

    # Create screening config
    config = ScreeningConfig()

    # Create full-text screener
    screener = FullTextScreener(config)

    # Test screening
    results = await screener.screen_fulltexts([study])

    print(f"Screening results: {len(results)}")
    for result in results:
        print(f"  Study: {result.study_id}")
        print(f"  Decision: {result.decision}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Reason: {result.reason}")

    return results


async def main():
    """Run all tests."""
    print("Running screening module tests...\n")

    try:
        # Test abstract screening
        abstract_results = await test_abstract_screening()

        # Test full-text screening
        fulltext_results = await test_fulltext_screening()

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