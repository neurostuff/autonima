#!/usr/bin/env python3
"""
Example script to run the Autonima systematic review pipeline.

This script demonstrates how to use the Autonima package to run
a complete systematic review pipeline from the command line.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the autonima package to the path if running from examples
sys.path.insert(0, str(Path(__file__).parent.parent))

from autonima.pipeline import run_pipeline_from_config


async def main():
    """Main function to run the pipeline."""
    # Use command line argument or default config file
    config_path = sys.argv[1] if len(sys.argv) > 1 else "examples/sample_config.yml"

    if not Path(config_path).exists():
        print(f"Error: Configuration file not found: {config_path}")
        print("Usage: python examples/run_pipeline.py [config_file]")
        print("Default: examples/sample_config.yml")
        sys.exit(1)

    try:
        print(f"Running Autonima pipeline with config: {config_path}")
        results = await run_pipeline_from_config(config_path)

        print("\nPipeline completed successfully!")
        print(f"Results saved to: {results.config.output.directory}")

        # Print summary statistics
        stats = results.execution_stats
        prisma_stats = stats.get('prisma_stats', {})
        print("\nSummary:")
        print(f"  Total studies identified: {prisma_stats.get('total_identified', 0)}")
        print(f"  Final studies included: {prisma_stats.get('final_included', 0)}")

        if results.errors:
            print(f"\nWarnings/Errors: {len(results.errors)}")
            for error in results.errors:
                print(f"  - {error}")

    except Exception as e:
        print(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Alternative: Use the CLI directly
    # from autonima.cli import main as cli_main
    # cli_main()

    asyncio.run(main())