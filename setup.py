"""Setup script for Autonima package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read version from package
def get_version():
    """Get version from package __init__.py."""
    init_file = this_directory / "autonima" / "__init__.py"
    if init_file.exists():
        for line in init_file.read_text().split('\n'):
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"\'')
    return "0.1.0"

setup(
    name="autonima",
    version=get_version(),
    author="Alejandro de la Vega",
    author_email="delavega@utexas.edu",
    license="MIT",
    description="LLM-powered automated systematic review and meta-analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/neurostuff/autonima",
    packages=find_packages(exclude=["tests*", "examples*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.10,<3.12",
    # Python 3.12 is NOT supported, for two independent reasons:
    #   * pubget caps joblib at <=1.1.1, and joblib 1.1.1 imports distutils,
    #     which 3.12 removed -- autonima.retrieval fails to import.
    #   * the scikit-learn pulled in by that same chain imports ComplexWarning
    #     from numpy.core.numeric, which numpy 2.0 removed.
    # Both are upstream. Relaxing pubget's joblib cap is the first thing to fix.
    install_requires=[
        "pyyaml>=6.0",
        "pydantic>=2.0",
        "click>=8.0",
        "requests>=2.28.0",
        "typing-extensions>=4.0",
        "biopython>=1.81",
        "pandas>=2.0",
        "pubget>=0.0.8",
        "beautifulsoup4>=4.9.0",
        "lxml>=4.6.0",
        # Imported at module scope by pipeline.py, screening/screener.py,
        # annotation/processor.py and coordinates/parser.py -- not optional.
        "tqdm>=4.60"
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-asyncio>=0.21.0",
            "black>=22.0",
            "flake8>=5.0",
            "mypy>=1.0",
            "pre-commit>=3.0",
        ],
        "llm": [
            "openai>=1.0"
        ],
        "readability": [
            "readabilipy>=0.2.0"
        ],
        "meta": [
            "nimare>=0.1.0"
        ],
        "docs": [
            "mkdocs>=1.6",
            "mkdocs-material>=9.5",
            "mkdocs-click>=0.8",
            "pymdown-extensions>=10.0",
        ],
        "ui": [
            "fastapi>=0.115",
            "uvicorn>=0.30",
        ],
    },
    entry_points={
        "console_scripts": [
            "autonima=autonima.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "autonima": ["py.typed"],
        "autonima.templates": ["*.yml"],
        "autonima.webui": ["static/*.html", "static/*.css", "static/*.jsx"],
    },
    keywords=[
        "systematic-review",
        "meta-analysis",
        "neuroimaging",
        "pubmed",
        "llm",
        "ai",
        "literature-search",
        "prisma",
        "evidence-synthesis"
    ],
    project_urls={
        "Documentation": "https://neurostuff.github.io/autonima/",
        "Source": "https://github.com/neurostuff/autonima",
        "Tracker": "https://github.com/neurostuff/autonima/issues",
    },
)
