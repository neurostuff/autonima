"""Coordinate parsing processor for the pipeline."""

import logging
from typing import List

from .openai_client import CoordinateParsingClient
from .prompts import create_coordinate_parsing_prompt

logger = logging.getLogger(__name__)


class CoordinateProcessor:
    """Processor for parsing coordinates from activation tables."""
    
    def __init__(self, model: str = "gpt-4o-mini", 
                 path_preference: List[str] = ['table_raw_path', 'table_data_path']):
        """
        Initialize the coordinate processor.
        
        Args:
            model: The model to use for parsing
        """
        self.model = model
        self.path_preference = path_preference
        self.client = CoordinateParsingClient()

    
    def process_single_table(self, table):
        """
        Process a single activation table and extract analyses.
        
        Args:
            table: The ActivationTable to process
            
        Returns:
            List of analyses extracted from the table
        """
        try:
            # Load the raw table content using the table's method
            table.load_raw_table()
            
            # If we couldn't load the raw table content, return empty list
            if table.raw_table is None:
                logger.warning(f"No valid table path found for table: {table.table_id}")
                return []
            
            # Use the raw_table content directly
            table_text = table.raw_table
            
            # Create a prompt for the table
            prompt = self._create_table_prompt(
                table_text,
                table_caption=table.table_caption or "",
                table_foot=table.table_foot or ""
            )
            
            # Parse the table
            result = self.client.parse_analyses(prompt, model=self.model)
            
            # Set the table_id for each analysis
            for analysis in result.analyses:
                analysis.table_id = table.table_id
            
            return result.analyses
            
        except Exception as e:
            logger.warning(f"Error processing table {table.table_id}: {e}")
            return []
    
    def _create_table_prompt(
        self,
        table_text: str,
        table_caption: str = "",
        table_foot: str = "",
    ) -> str:
        """
        Create a prompt for parsing a table.
        
        Args:
            table_text: The text content of the table
            table_caption: The caption of the table
            table_foot: The footer of the table
            
        Returns:
            The prompt for the LLM
        """
        return create_coordinate_parsing_prompt(
            table_text,
            table_caption=table_caption,
            table_foot=table_foot,
        )
