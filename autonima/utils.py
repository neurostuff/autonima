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