"""Command-line interface for Autonima."""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

# Note: This would be installed as a dependency
# import click

from .config import ConfigManager, ConfigurationError
from .pipeline import run_pipeline_from_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Mock click functionality for now (would use actual click in real implementation)
class MockClick:
    """Mock click functionality for development."""

    @staticmethod
    def command():
        def decorator(func):
            return func
        return decorator

    @staticmethod
    def option(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    @staticmethod
    def argument(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


# Use mock click for now
click = MockClick()


@click.command()
@click.option('--config', '-c', required=True,
              help='Path to YAML configuration file')
@click.option('--output', '-o', default=None,
              help='Output directory (overrides config)')
@click.option('--verbose', '-v', is_flag=True,
              help='Enable verbose logging')
@click.option('--dry-run', is_flag=True,
              help='Validate configuration without running pipeline')
def run(config: str, output: Optional[str], verbose: bool, dry_run: bool):
    """
    Run the Autonima systematic review pipeline.

    This command executes the complete systematic review workflow:
    1. Literature search via PubMed
    2. Abstract screening with LLMs
    3. Full-text retrieval
    4. Full-text screening
    5. Output generation with PRISMA compliance

    Example:
        autonima run --config config.yaml
        autonima run --config config.yaml --verbose
        autonima run --config config.yaml --dry-run
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled")

    config_path = Path(config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    try:
        # Load and validate configuration
        logger.info(f"Loading configuration from: {config_path}")
        config_manager = ConfigManager()
        pipeline_config = config_manager.load_from_file(str(config_path))

        if output:
            pipeline_config.output.directory = output

        logger.info(f"Pipeline objective: {pipeline_config.objective}")
        logger.info(f"Search database: {pipeline_config.search.database}")
        logger.info(f"Search query: {pipeline_config.search.query}")
        logger.info(f"Output directory: {pipeline_config.output.directory}")

        if dry_run:
            logger.info("Dry run completed successfully - configuration is valid")
            return

        # Run the pipeline
        logger.info("Starting pipeline execution...")

        async def execute_pipeline():
            results = await run_pipeline_from_config(str(config_path))

            # Print summary
            stats = results.execution_stats
            prisma_stats = stats.get('prisma_stats', {})

            print("\n" + "="*60)
            print("AUTONIMA PIPELINE COMPLETED")
            print("="*60)
            print(f"Objective: {pipeline_config.objective}")
            print(f"Total studies identified: {prisma_stats.get('total_identified', 0)}")
            print(f"Studies included: {prisma_stats.get('final_included', 0)}")
            print(f"Output directory: {pipeline_config.output.directory}")

            if results.errors:
                print(f"\nErrors encountered: {len(results.errors)}")
                for error in results.errors:
                    print(f"  - {error}")

            print("\nResults saved to:", pipeline_config.output.directory)
            return results

        # Run the async pipeline
        results = asyncio.run(execute_pipeline())

    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)


@click.command()
@click.option('--config', '-c', required=True,
              help='Path to YAML configuration file')
def validate(config: str):
    """
    Validate a configuration file without running the pipeline.

    This command checks if the configuration file is valid and
    all required parameters are present.

    Example:
        autonima validate --config config.yaml
    """
    config_path = Path(config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    try:
        config_manager = ConfigManager()
        pipeline_config = config_manager.load_from_file(str(config_path))

        print(f"✓ Configuration file is valid: {config_path}")
        print(f"✓ Objective: {pipeline_config.objective}")
        print(f"✓ Search database: {pipeline_config.search.database}")
        print(f"✓ Search query: {pipeline_config.search.query}")
        print(f"✓ Inclusion criteria: {len(pipeline_config.inclusion_criteria)}")
        print(f"✓ Exclusion criteria: {len(pipeline_config.exclusion_criteria)}")
        print(f"✓ Output directory: {pipeline_config.output.directory}")

    except ConfigurationError as e:
        logger.error(f"Configuration validation failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during validation: {e}")
        sys.exit(1)


@click.command()
def create_sample_config():
    """
    Create a sample configuration file.

    This command generates a sample configuration file that can be
    used as a starting point for your systematic review.

    Example:
        autonima create-sample-config > config.yaml
    """
    try:
        from .config import create_sample_config_file
        import tempfile

        # Create sample config in temporary file
        temp_path = Path(tempfile.mktemp(suffix='.yaml'))
        create_sample_config_file(str(temp_path))

        # Read and print the sample config
        with open(temp_path, 'r') as f:
            sample_config = f.read()

        print("# Sample Autonima Configuration")
        print("# Copy this to a file and modify as needed")
        print()
        print(sample_config)

        # Clean up
        temp_path.unlink()

    except Exception as e:
        logger.error(f"Failed to create sample config: {e}")
        sys.exit(1)


# CLI group would be used with actual click
# @click.group()
# def cli():
#     """Autonima: Automated Neuroimaging Meta-Analysis"""
#     pass


# cli.add_command(run)
# cli.add_command(validate)
# cli.add_command(create_sample_config)


def main():
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        print("Autonima: Automated Neuroimaging Meta-Analysis")
        print()
        print("Usage:")
        print("  python -m autonima run --config <config_file>")
        print("  python -m autonima validate --config <config_file>")
        print("  python -m autonima create-sample-config")
        print()
        print("For help: python -m autonima <command> --help")
        sys.exit(1)

    command = sys.argv[1]

    if command == "run":
        # Simple argument parsing for run command
        config_path = None
        verbose = False
        dry_run = False

        i = 2
        while i < len(sys.argv):
            if sys.argv[i] == "--config" and i + 1 < len(sys.argv):
                config_path = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == "--verbose":
                verbose = True
                i += 1
            elif sys.argv[i] == "--dry-run":
                dry_run = True
                i += 1
            else:
                i += 1

        if not config_path:
            print("Error: --config is required")
            sys.exit(1)

        # Call the run function
        run(config_path, None, verbose, dry_run)

    elif command == "validate":
        # Simple argument parsing for validate command
        config_path = None

        i = 2
        while i < len(sys.argv):
            if sys.argv[i] == "--config" and i + 1 < len(sys.argv):
                config_path = sys.argv[i + 1]
                i += 2
            else:
                i += 1

        if not config_path:
            print("Error: --config is required")
            sys.exit(1)

        validate(config_path)

    elif command == "create-sample-config":
        create_sample_config()

    else:
        print(f"Unknown command: {command}")
        print("Available commands: run, validate, create-sample-config")
        sys.exit(1)


if __name__ == "__main__":
    main()