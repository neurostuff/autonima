"""NiMADS data models for neuroimaging meta-analysis."""

from dataclasses import dataclass, field
from typing import List, Optional, Union
from .schema import Analysis, CoordinatePoint


@dataclass
class PointValue:
    """Represents a value associated with a coordinate point."""
    kind: Optional[str] = None
    value: Optional[Union[float, str]] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "kind": self.kind,
            "value": self.value
        }


@dataclass
class Point:
    """A stereotaxic 3 dimensional coordinate describing a reported coordinate."""
    coordinates: List[float]
    space: Optional[str] = None
    kind: Optional[str] = None
    label_id: Optional[str] = None
    values: List[PointValue] = field(default_factory=list)
    analysis_id: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "coordinates": self.coordinates,
            "space": self.space,
            "kind": self.kind,
            "label_id": self.label_id,
            "values": [value.to_dict() for value in self.values],
            "analysis_id": self.analysis_id
        }


@dataclass
class Condition:
    """A representative term for a psychological, pharmacological, medical, or physical state."""
    name: Optional[str] = None
    description: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description
        }


@dataclass
class Image:
    """Statistical image (e.g., beta, t-statistic, and/or z-statistic image)."""
    id: Optional[str] = None
    filename: Optional[str] = None
    space: Optional[str] = None
    kind: Optional[str] = None
    analysis_id: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "filename": self.filename,
            "space": self.space,
            "kind": self.kind,
            "analysis_id": self.analysis_id
        }


@dataclass
class NimadsAnalysis:
    """A contrast of weighted conditions with associated statistical results."""
    id: str
    name: Optional[str] = None
    description: Optional[str] = None
    weights: List[float] = field(default_factory=list)
    conditions: List[Condition] = field(default_factory=list)
    images: List[Image] = field(default_factory=list)
    points: List[Point] = field(default_factory=list)
    study_id: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "weights": self.weights,
            "conditions": [condition.to_dict() for condition in self.conditions],
            "images": [image.to_dict() for image in self.images],
            "points": [point.to_dict() for point in self.points],
            "study_id": self.study_id
        }


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
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "doi": self.doi,
            "name": self.name,
            "metadata": self.metadata,
            "description": self.description,
            "publication": self.publication,
            "pmid": self.pmid,
            "authors": self.authors,
            "year": self.year,
            "analyses": [analysis.to_dict() for analysis in self.analyses]
        }


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
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "publication": self.publication,
            "doi": self.doi,
            "pmid": self.pmid,
            "studies": [study.to_dict() for study in self.studies]
        }


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


@dataclass
class NoteCollection:
    """The storage object for all notes within an annotation for a single analysis."""
    note: Optional[dict] = None
    analysis_id: Optional[str] = None
    annotation_id: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "note": self.note,
            "analysis": self.analysis_id,
            "annotation": self.annotation_id
        }


@dataclass
class Annotation:
    """An annotation describes each analysis within a studyset with typically subjective information."""
    id: str
    name: Optional[str] = None
    description: Optional[str] = None
    metadata: Optional[dict] = None
    note_keys: Optional[dict] = None
    studyset_id: Optional[str] = None
    notes: List[NoteCollection] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "metadata": self.metadata,
            "note_keys": self.note_keys,
            "studyset": self.studyset_id,
            "notes": [note.to_dict() for note in self.notes]
        }


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


def create_default_annotation(studyset_id: str, studyset: Studyset) -> Annotation:
    """Create a default annotation with include=True for all analyses in the studyset."""
    annotation_id = f"annotation_{studyset_id}"
    annotation = Annotation(
        id=annotation_id,
        name="replication_annotations",
        description="",
        note_keys={"include": "boolean"},
        studyset_id=studyset_id
    )
    
    # Create notes for each analysis
    for study in studyset.studies:
        for analysis in study.analyses:
            note = NoteCollection(
                note={"include": True},
                analysis_id=analysis.id,
                annotation_id=annotation_id
            )
            annotation.notes.append(note)
    
    return annotation