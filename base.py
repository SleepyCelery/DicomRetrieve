from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, INTEGER
import json

Base = declarative_base()


class BaseDescription(Base):
    __abstract__ = True
    IndexID = Column(INTEGER, nullable=False, primary_key=True, autoincrement=True, index=True)
    SeriesInstanceUID = Column(String(64), unique=True, nullable=False)
    PatientName = Column(String(16), nullable=False)
    PatientSex = Column(String(16), nullable=False)
    PatientBirthDate = Column(String(32), nullable=False)
    PatientAge = Column(String(16))
    AcquisitionNumber = Column(INTEGER, nullable=False)
    ProtocolName = Column(String(64), nullable=False, default="")
    StudyDate = Column(String(16), nullable=False)
    StudyTime = Column(String(16), nullable=False)
    InstitutionName = Column(String(64))

    def __repr__(self):
        return json.dumps({
            'SeriesInstanceUID': self.SeriesInstanceUID,
            'PatientName': self.PatientName,
            'PatientSex': self.PatientSex,
            'PatientBirthDate': self.PatientBirthDate,
            'PatientAge': self.PatientAge,
            'AcquisitionNumber': str(self.AcquisitionNumber),
            'ProtocolName': self.ProtocolName,
            'StudyDate': self.StudyDate,
            'StudyTime': self.StudyTime,
            'InstitutionName': self.InstitutionName
        })


class DicomFileSavingPath(Base):
    __tablename__ = "DicomFileSavingPath"
    SeriesSequenceID = Column(String(64), primary_key=True)
    RelativePath = Column(String(256), nullable=False)

    def __repr__(self):
        return self.SeriesSequenceID


class LumbarDiscDescription(BaseDescription):
    """
    腰椎间盘断层
    """
    __tablename__ = "LumbarDiscDescription"
    ProtocolName = Column(String(64), nullable=False, default="腰椎间盘断层")


class SearchResult:
    def __init__(self, SeriesRecord: BaseDescription, Distance: float):
        self.SeriesRecord = SeriesRecord
        self.Distance = Distance

    def __lt__(self, other):
        return self.Distance < other.Distance

    def __gt__(self, other):
        return self.Distance > other.Distance

    def __eq__(self, other):
        return self.Distance == other.Distance

    def __repr__(self):
        return str({**json.loads(str(self.SeriesRecord)), 'Distance': str(self.Distance)})

    def to_dict(self):
        return {**json.loads(str(self.SeriesRecord)), 'Distance': str(self.Distance)}
