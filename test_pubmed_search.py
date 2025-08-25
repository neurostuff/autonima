#!/usr/bin/env python3
"""
Test script to verify PubMed search functionality.
"""

import asyncio
import sys
from pathlib import Path

# Add the autonima package to the path
sys.path.insert(0, str(Path(__file__).parent))

from autonima.models.types import SearchConfig
from autonima.search.pubmed import PubMedSearch


async def test_pubmed_search():
    """Test PubMed search functionality."""
    print("Testing PubMed search functionality...")
    
    # Create a search configuration
    config = SearchConfig(
        database="pubmed",
        query="fMRI AND schizophrenia",
        max_results=5,
        email="test@example.com"  # NCBI requires an email for API usage
    )
    
    # Initialize the PubMed search engine
    search_engine = PubMedSearch(config)
    
    print("Search configuration:")
    print(f"  Database: {config.database}")
    print(f"  Query: {config.query}")
    print(f"  Max results: {config.max_results}")
    print(f"  Email: {config.email}")
    
    try:
        # Execute the search
        print("\nExecuting search...")
        studies = await search_engine.search(config.query)
        
        print("\nSearch completed successfully!")
        print(f"Found {len(studies)} studies")
        
        # Display information about the found studies
        if studies:
            print("\nFirst few studies found:")
            for i, study in enumerate(studies[:3]):  # Show first 3 studies
                print(f"\nStudy {i+1}:")
                print(f"  PMID: {study.pmid}")
                print(f"  Title: {study.title}")
                authors_str = ", ".join(study.authors[:3])
                if len(study.authors) > 3:
                    authors_str += "..."
                print(f"  Authors: {authors_str}")
                print(f"  Journal: {study.journal}")
                print(f"  Publication Date: {study.publication_date}")
                print(f"  DOI: {study.doi}")
                keywords_str = ", ".join(study.keywords[:5])
                if len(study.keywords) > 5:
                    keywords_str += "..."
                print(f"  Keywords: {keywords_str}")
                if study.abstract:
                    print(f"  Abstract preview: {study.abstract[:100]}...")
        
        # Test search info
        search_info = search_engine.get_search_info()
        print("\nSearch engine info:")
        print(f"  Engine: {search_info['engine']}")
        print(f"  API URL: {search_info['api_url']}")
        print(f"  Max results: {search_info['max_results']}")
        
        print("\n✅ PubMed search test passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ PubMed search test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    print("=" * 60)
    print("PUBMED SEARCH TEST")
    print("=" * 60)
    
    success = await test_pubmed_search()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 PUBMED SEARCH TEST PASSED!")
        print("PubMed search implementation is working correctly.")
    else:
        print("❌ PUBMED SEARCH TEST FAILED!")
        print("There were issues with the PubMed search implementation.")
    print("=" * 60)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)