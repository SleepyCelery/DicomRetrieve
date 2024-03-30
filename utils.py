import zipfile
from tempfile import TemporaryDirectory
import os
import shutil


def zip2dicom_dir(zipfile_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    with TemporaryDirectory() as tmpdir:
        z = zipfile.ZipFile(zipfile_path, 'r')
        z.extractall(path=tmpdir)
        for root, dirs, files in os.walk(tmpdir):
            for file in files:
                if os.path.splitext(file)[-1] == '.dcm':
                    shutil.copyfile(os.path.join(root, file), os.path.join(output_path, file))


def dicom_files2zip(dicom_files: list, zip_output_path):
    with zipfile.ZipFile(file=zip_output_path, mode='w') as zf:
        for path in dicom_files:
            zf.write(filename=path, arcname=os.path.split(path)[-1])


if __name__ == '__main__':
    with TemporaryDirectory() as tmpdir:
        zip2dicom_dir('test.zip', tmpdir)
        print(os.listdir(tmpdir))
