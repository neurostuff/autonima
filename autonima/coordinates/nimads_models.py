"""NiMADS data models for neuroimaging meta-analysis."""

from dataclasses import dataclass, field
from typing import List, Optional, Union
from .schema import Analysis, CoordinatePoint


@dataclass
class PointValue:
    """Represents a value associated with a coordinate point."""
    kind: Optional[str] = None
    value: Optional[Union[float, str]] = None


@dataclass
class Point:
    """A stereotaxic 3 dimensional coordinate describing a reported coordinate."""
    coordinates: List[float]
    space: Optional[str] = None
    kind: Optional[str] = None
    label_id: Optional[str] = None
    values: List[PointValue] = field(default_factory=list)
    analysis_id: Optional[str] = None


@dataclass
class NimadsAnalysis:
    """A contrast of weighted conditions with associated statistical results."""
    id: str
    name: Optional[str] = None
    description: Optional[str] = None
    weights: List[float] = field(default_factory=list)
    points: List[Point] = field(default_factory=list)
    study_id: Optional[str] = None


@dataclass
class Study:
    """A publishable unit of research."""
    id: str
    doi: Optional[str] = None
    name: Optional[str] = None
    metadata: Optional[dict] = None
    description: Optional[str] = None
    publication: Optional[str] = None
    pmid: Optional[str] = None
    authors: Optional[str] = None
    year: Optional[int] = None
    analyses: List[NimadsAnalysis] = field(default_factory=list)


@dataclass
class Studyset:
    """A collection of studies (e.g., publications)."""
    id: str
    name: Optional[str] = None
    description: Optional[str] = None
    publication: Optional[str] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    studies: List[Study] = field(default_factory=list)


def convert_to_nimads_point(analysis_id: str, point: CoordinatePoint) -> Point:
    """Convert a coordinate point to a NiMADS point."""
    nimads_point = Point(
        coordinates=point.coordinates,
        space=point.space,
        analysis_id=analysis_id
    )
    
    # Convert point values if they exist
    if point.values:
        for value in point.values:
            # Preserve the original value
            nimads_point.values.append(
                PointValue(
                    kind=value.kind,
                    value=value.value
                )
            )
    
    return nimads_point


def convert_to_nimads_analysis(analysis_id: str, analysis: Analysis, study_id: str) -> NimadsAnalysis:
    """Convert an analysis to a NiMADS analysis."""
    nimads_analysis = NimadsAnalysis(
        id=analysis_id,
        name=analysis.name,
        description=analysis.description,
        study_id=study_id
    )
    
    # Convert points
    for point in analysis.points:
        nimads_point = convert_to_nimads_point(analysis_id, point)
        nimads_analysis.points.append(nimads_point)
    
    return nimads_analysis


def convert_to_nimads_study(study_id: str, autonima_study: 'autonima.models.types.Study') -> Study:
    """Convert an autonima study to a NiMADS study."""
    # Extract year from publication date if available
    year = None
    if autonima_study.publication_date:
        try:
            # Try to extract year from YYYY-MM-DD format
            year = int(autonima_study.publication_date.split('-')[0])
        except (ValueError, IndexError):
            pass
    
    nimads_study = Study(
        id=study_id,
        doi=autonima_study.doi,
        name=autonima_study.title,
        description=autonima_study.abstract,
        publication=autonima_study.journal,
        pmid=autonima_study.pmid,
        authors=', '.join(autonima_study.authors) if autonima_study.authors else None,
        year=year
    )
    
    # Convert analyses
    for i, analysis in enumerate(autonima_study.analyses):
        analysis_id = f"{study_id}_analysis_{i}"
        nimads_analysis = convert_to_nimads_analysis(analysis_id, analysis, study_id)
        nimads_study.analyses.append(nimads_analysis)
    
    return nimads_study


def convert_to_nimads_studyset(studyset_id: str, studies: List['autonima.models.types.Study'], name: Optional[str] = None) -> Studyset:
    """Convert a list of autonima studies to a NiMADS studyset."""
    studyset = Studyset(
        id=studyset_id,
        name=name or "Autonima Generated Studyset"
    )
    
    # Convert studies
    for i, study in enumerate(studies):
        study_id = f"study_{i}"
        nimads_study = convert_to_nimads_study(study_id, study)
        studyset.studies.append(nimads_study)
    
    return studyset