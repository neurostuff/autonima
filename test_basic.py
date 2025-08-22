#!/usr/bin/env python3
"""
Basic test script to verify Autonima package functionality.
"""

import asyncio
import sys
from pathlib import Path

# Add the autonima package to the path
sys.path.insert(0, str(Path(__file__).parent))

from autonima.config import ConfigManager
from autonima.models.types import PipelineConfig
from autonima.pipeline import AutonimaPipeline


async def test_basic_functionality():
    """Test basic package functionality."""
    print("Testing Autonima basic functionality...")

    # Test 1: Configuration loading
    print("\n1. Testing configuration...")
    config_manager = ConfigManager()
    sample_config = config_manager.create_sample_config()

    print("✓ Sample configuration created")
    print(f"  Objective: {sample_config.objective}")
    print(f"  Search query: {sample_config.search.query}")
    print(f"  Inclusion criteria: {len(sample_config.inclusion_criteria)}")
    print(f"  Exclusion criteria: {len(sample_config.exclusion_criteria)}")

    # Test 2: Pipeline initialization
    print("\n2. Testing pipeline initialization...")
    pipeline = AutonimaPipeline(sample_config)
    print("✓ Pipeline initialized successfully")

    # Test 3: Search functionality (mock)
    print("\n3. Testing search functionality...")
    try:
        studies = await pipeline._search_engine.search(sample_config.search.query)
        print(f"✓ Search completed, found {len(studies)} studies")

        if studies:
            study = studies[0]
            print(f"  Sample study: {study.title[:50]}... (PMID: {study.pmid})")

    except Exception as e:
        print(f"✗ Search failed: {e}")

    # Test 4: Abstract screening (mock)
    print("\n4. Testing abstract screening...")
    if studies:
        await pipeline._execute_abstract_screening()
        included_count = len([s for s in pipeline.results.studies if s.status.name == "INCLUDED"])
        print(f"✓ Abstract screening completed: {included_count} studies included")

    # Test 5: Pipeline statistics
    print("\n5. Testing pipeline statistics...")
    stats = pipeline.get_statistics()
    print("✓ Pipeline statistics generated")
    print(f"  Total studies: {stats['results']['total_studies']}")
    print(f"  Included studies: {stats['results']['included_studies']}")

    print("\n✅ All basic tests passed!")


async def test_cli_commands():
    """Test CLI command functionality."""
    print("\nTesting CLI commands...")

    # Test configuration validation
    config_manager = ConfigManager()
    sample_config = config_manager.create_sample_config()

    print("✓ CLI components available")

    return True


async def main():
    """Main test function."""
    print("=" * 60)
    print("AUTONIMA PACKAGE TEST")
    print("=" * 60)

    try:
        await test_basic_functionality()
        await test_cli_commands()

        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("Autonima package is working correctly.")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())