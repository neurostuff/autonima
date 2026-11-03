"""Utility functions for Autonima."""

# Global debug flag
DEBUG_MODE = False


def set_debug_mode(debug: bool):
    """
    Set the global debug mode flag.
    
    Args:
        debug: Whether to enable debug mode
    """
    global DEBUG_MODE
    DEBUG_MODE = debug


def atomic_write_json(file_path, data):
    """Write JSON atomically to avoid partially written output files."""
    import json
    import os

    file_path.parent.mkdir(parents=True, exist_ok=True)
    temp_file = file_path.with_suffix(file_path.suffix + ".tmp")
    with open(temp_file, 'w') as f:
        json.dump(data, f, indent=2)
    os.replace(temp_file, file_path)


def log_error_with_debug(logger, message):
    """
    Log an error message and enter pdb post-mortem if debug mode is enabled.
    
    Args:
        logger: Logger instance to use for logging
        message: Error message to log
    """
    logger.error(message)
    
    if DEBUG_MODE:
        import pdb
        import sys
        # Get the current exception info
        exc_info = sys.exc_info()
        if exc_info[0] is not None:
            pdb.post_mortem(exc_info[2])