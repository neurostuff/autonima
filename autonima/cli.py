"""Command-line interface for Autonima."""

import asyncio
import logging
import sys
from pathlib import Path

import click

from .config import ConfigManager, ConfigurationError
from .pipeline import run_pipeline_from_config
from .utils import set_debug_mode, log_error_with_debug

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Silence verbose HTTP request logs from httpx (used by OpenAI client)
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


@click.command()
@click.argument('config', type=click.Path(exists=True))
@click.argument('output_folder', type=click.Path())
@click.option('--verbose', '-v', is_flag=True,
              help='Enable verbose logging')
@click.option('--dry-run', is_flag=True,
              help='Validate configuration without running pipeline')
@click.option('--debug', is_flag=True,
              help='Enable debug mode with post-mortem debugging on errors')
@click.option('--num-workers', '-j', type=int, default=1,
              help='Number of parallel workers for screening (default: 1)')
def run(
    config: str,
    output_folder: str,
    verbose: bool,
    dry_run: bool,
    debug: bool,
    num_workers: int
):
    """
    Run the Autonima systematic review pipeline.

    This command executes the complete systematic review workflow:
    1. Literature search via PubMed
    2. Abstract screening with LLMs
    3. Full-text retrieval
    4. Full-text screening
    5. Output generation with PRISMA compliance

    Arguments:
        CONFIG          Path to YAML configuration file
        OUTPUT_FOLDER   Output folder for all results and intermediary files

    Options:
        -v, --verbose   Enable verbose logging
        --dry-run       Validate configuration without running pipeline

    Examples:
        autonima run config.yaml results
        autonima run config.yaml results --verbose
        autonima run config.yaml results --dry-run
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled")
    
    # Set debug mode globally
    set_debug_mode(debug)

    config_path = Path(config)
    if not config_path.exists():
        log_error_with_debug(
            logger, f"Configuration file not found: {config_path}"
        )
        sys.exit(1)

    try:
        # Load and validate configuration
        logger.info(f"Loading configuration from: {config_path}")
        config_manager = ConfigManager()
        pipeline_config = config_manager.load_from_file(str(config_path))

        # Set output directory to the specified output folder
        pipeline_config.output.directory = output_folder

        if dry_run:
            logger.info("Dry run completed successfully")
            logger.info("Configuration is valid")
            return

        async def execute_pipeline():
            results = await run_pipeline_from_config(
                config=pipeline_config, num_workers=num_workers
            )

            # Print summary
            stats = results.execution_stats
            prisma_stats = stats.get('prisma_stats', {})

            print("\n" + "="*60)
            print("AUTONIMA PIPELINE COMPLETED")
            print("="*60)
            identified = prisma_stats.get('total_identified', 0)
            print(f"Total studies identified: {identified}")
            print(f"Studies included: {prisma_stats.get('final_included', 0)}")
            print(f"Output directory: {pipeline_config.output.directory}")

            if results.errors:
                print(f"\nErrors encountered: {len(results.errors)}")
                for error in results.errors:
                    print(f"  - {error}")

            print("\nResults saved to:", pipeline_config.output.directory)
            return results

        # Run the async pipeline
        asyncio.run(execute_pipeline())

    except ConfigurationError as e:
        log_error_with_debug(logger, f"Configuration error: {e}")
        if debug:
            import pdb
            pdb.post_mortem()
        sys.exit(1)
    except Exception as e:
        log_error_with_debug(logger, f"Pipeline execution failed: {e}")
        if debug:
            import pdb
            pdb.post_mortem()
        sys.exit(1)


@click.command()
@click.argument('config', type=click.Path(exists=True))
@click.argument('output_folder', type=click.Path())
@click.option('--debug', is_flag=True,
              help='Enable debug mode with post-mortem debugging on errors')
def validate(config: str, output_folder: str, debug: bool):
    """
    Validate a configuration file without running the pipeline.

    This command checks if the configuration file is valid and
    all required parameters are present.

    Arguments:
        CONFIG          Path to YAML configuration file
        OUTPUT_FOLDER   Output folder for all results and intermediary files

    Examples:
        autonima validate config.yaml results
    """
    config_path = Path(config)
    if not config_path.exists():
        log_error_with_debug(
            logger, f"Configuration file not found: {config_path}"
        )
        sys.exit(1)

    try:
        config_manager = ConfigManager()
        pipeline_config = config_manager.load_from_file(str(config_path))
        
        # Set output directory to the specified output folder
        pipeline_config.output.directory = output_folder

        print(f"✓ Configuration file is valid: {config_path}")
        print(f"✓ Search database: {pipeline_config.search.database}")
        print(f"✓ Search query: {pipeline_config.search.query}")
        
        # Print abstract screening info if available
        abstract_config = pipeline_config.screening.abstract
        abstract_objective = abstract_config.get('objective')
        abstract_inclusion = abstract_config.get('inclusion_criteria', [])
        abstract_exclusion = abstract_config.get('exclusion_criteria', [])
        
        if abstract_objective:
            print(f"✓ Abstract screening objective: {abstract_objective}")
        if abstract_inclusion:
            print(f"✓ Abstract inclusion criteria: {len(abstract_inclusion)}")
        if abstract_exclusion:
            print(f"✓ Abstract exclusion criteria: {len(abstract_exclusion)}")
        
        # Print fulltext screening info if available
        fulltext_config = pipeline_config.screening.fulltext
        fulltext_objective = fulltext_config.get('objective')
        fulltext_inclusion = fulltext_config.get('inclusion_criteria', [])
        fulltext_exclusion = fulltext_config.get('exclusion_criteria', [])
        
        if fulltext_objective:
            print(f"✓ Fulltext screening objective: {fulltext_objective}")
        if fulltext_inclusion:
            print(f"✓ Fulltext inclusion criteria: {len(fulltext_inclusion)}")
        if fulltext_exclusion:
            print(f"✓ Fulltext exclusion criteria: {len(fulltext_exclusion)}")
        
        print(f"✓ Output directory: {pipeline_config.output.directory}")

    except ConfigurationError as e:
        log_error_with_debug(logger, f"Configuration validation failed: {e}")
        if debug:
            import pdb
            pdb.post_mortem()
        sys.exit(1)
    except Exception as e:
        log_error_with_debug(
            logger, f"Unexpected error during validation: {e}"
        )
        if debug:
            import pdb
            pdb.post_mortem()
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
        log_error_with_debug(logger, f"Failed to create sample config: {e}")


# CLI group would be used with actual click
@click.group()
def cli():
    """Autonima: Automated Neuroimaging Meta-Analysis"""
    pass


cli.add_command(run)
cli.add_command(validate)
cli.add_command(create_sample_config)


def main():
    """Main CLI entry point."""
    cli()


if __name__ == "__main__":
    main()