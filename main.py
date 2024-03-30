import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from utils import *
from data_operations import *
from tempfile import TemporaryDirectory
import read_dicom
from model_backend import *
from starlette.background import BackgroundTask
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def build_response_json(status_code: int, description: str, message=None):
    """
    :param status_code: 只能为0或者1，0表示成功，1表示失败
    :param description: 描述字符串，成功时内容为success，失败时为失败的说明文本
    :param message: 要返回给浏览器的信息
    :return: 最终返回给浏览器的json响应，使用JSONResponse封装
    """
    if not (status_code == 0 or status_code == 1):
        raise ValueError('status code must be integer 0 or 1')
    if message is None:
        message = {}
    if status_code == 0:
        return JSONResponse({'status_code': status_code, 'description': description, 'message': message},
                            status_code=200)
    else:
        return JSONResponse({'status_code': status_code, 'description': description, 'message': message},
                            status_code=400)


@app.post("/upload_zip_file")
async def upload_zip_file(tomography: str, topn: int, file: UploadFile = File(...)):
    filename = file.filename
    if os.path.splitext(filename)[-1] != '.zip':
        return build_response_json(1, '上传的文件只能为zip格式')
    if tomography not in list(type_config.keys()):
        return build_response_json(1, 'tomography参数只能为{}'.format(','.join(list(type_config.keys()))))
    if topn < 0 or topn > 20:
        return build_response_json(1, 'top参数不能为负数或大于20')
    with TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "temp.zip"), mode='wb') as tmpfile:
            tmpfile.write(await file.read())
        zip2dicom_dir(os.path.join(tmpdir, "temp.zip"), os.path.join(tmpdir, 'dicomfiles'))
        series_id = read_dicom.read_series_in_dir(os.path.join(tmpdir, 'dicomfiles'))
        if len(series_id) != 1:
            return build_response_json(1, '仅支持上传包含单个Dicom序列的zip文件')
        files_list = os.listdir(os.path.join(tmpdir, 'dicomfiles'))
        if len(files_list) != 4:
            return build_response_json(1, '仅支持上传包含4帧的Dicom序列')

        message = {}
        temp_tags_dict = read_specific_tags(os.path.join(tmpdir, 'dicomfiles', files_list[0]), list(need_tags.keys()))
        message['upload_dicom_info'] = {
            need_tags['0020|000e']: temp_tags_dict['0020|000e'],
            need_tags['0010|0010']: temp_tags_dict['0010|0010'],
            need_tags['0010|0040']: temp_tags_dict['0010|0040'],
            need_tags['0010|0030']: temp_tags_dict['0010|0030'],
            need_tags['0010|1010']: temp_tags_dict['0010|1010'],
            need_tags['0020|0012']: temp_tags_dict['0020|0012'],
            need_tags['0018|1030']: temp_tags_dict['0018|1030'],
            need_tags['0008|0020']: temp_tags_dict['0008|0020'],
            need_tags['0008|0030']: temp_tags_dict['0008|0030'],
            need_tags['0008|0080']: temp_tags_dict['0008|0080']
        }
        model = load_model(tomography)
        image_array = read_dicom_dir(os.path.join(tmpdir, 'dicomfiles'))
        feature_vector = get_feature_vector(model, image_array)
        results = search_similar_topn(feature_vector, topn, tomography)
        message['search_similarity_results'] = [result.to_dict() for result in results]
        return build_response_json(0, 'success', message)


@app.get("/download_dicom_zip")
async def download_dicom_zip(series_id: str):
    paths = query_saving_path_by_series_id(series_id)
    if not paths:
        return build_response_json(1, f'数据库中没有Series ID为{series_id}的记录')
    dicom_files2zip(paths, f"{series_id}.zip")

    def rm_file(file_path):
        if os.path.exists(file_path):
            os.remove(file_path)

    task = BackgroundTask(rm_file, file_path=f"{series_id}.zip")
    return FileResponse(f"{series_id}.zip", filename=f"{series_id}.zip", background=task)


if __name__ == '__main__':
    uvicorn.run(app="main:app", host="0.0.0.0", port=5231, reload=True)
