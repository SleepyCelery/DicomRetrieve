# ---------- constants ---------- #
need_tags = {
    '0020|000e': 'SeriesInstanceUID',
    '0010|0010': 'PatientName',
    '0010|0040': 'PatientSex',
    '0010|0030': 'PatientBirthDate',
    '0010|1010': 'PatientAge',
    '0020|0012': 'AcquisitionNumber',
    '0018|1030': 'ProtocolName',
    '0008|0020': 'StudyDate',
    '0008|0030': 'StudyTime',
    '0008|0080': 'InstitutionName',
    '0020|0013': 'InstanceNumber'
}

# ---------- tomography type configs ---------- #
from base import *
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def create_meta_session(database_url, expire_on_commit: bool = False):
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=expire_on_commit)


meta_session = create_meta_session('sqlite:///DicomRetrieve.db', expire_on_commit=False)

type_config = {
    'LumbarDisc':
        {
            'index_file': 'LumbarDisc.index',
            'model_file': 'model_resnet34.pth',
            'feature_vector_length': 128
        }
}
