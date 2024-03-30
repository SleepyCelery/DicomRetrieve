"""
离线处理流程：
1. 读取目标目录下的所有dcm文件，并提取其需要的tags，拼接文件的相对路径，并读取或创建新的Faiss Index（判断是否存在索引文件）
2. 构建描述表和路径表对应的数据行，添加到数据库，并将成功存入数据库的所有数据对象保存到列表中
3. 添加完目录下的所有文件后，对列表中的SeriesID进行迭代，读取其AcquisitionNumber和SeriesID以及自增的IndexID
4. 拼接所有的dcm文件路径表主键，读取目录信息，然后移动到临时目录，此时临时目录内都是同一个Series的Dicom文件
5. 对目录下所有Dicom文件进行读取、图像预处理，并使用模型推理出特征向量
6. 向Faiss索引中添加带id的记录，id为数据库内自增的IndexID
7. 提交数据库、保存Faiss索引

注意事项：
1. 由于faiss的删除向量时间复杂度为O(n)，故不提供删除功能，请直接使用delete_by_series_id函数在数据库中删除记录后重建faiss索引，经测试，在RTX2060上，可以达到每秒钟25个Dicom序列的重建
"""
import os
import faiss
from model_backend import read_dicom_dir, load_model, get_feature_vector
from read_dicom import read_specific_tags
from tempfile import TemporaryDirectory
from config import *
from loguru import logger
import shutil
import numpy as np
from tqdm import tqdm


def build_from_dir(target_dir, tomography_type):
    if tomography_type == 'LumbarDisc':
        DescriptionObj = LumbarDiscDescription
        index_file = type_config[tomography_type]['index_file']
        feature_vector_length = type_config[tomography_type]['feature_vector_length']
        logger.info(f'断层扫描类型为：{tomography_type}')
    else:
        raise ValueError('The file_type parameter must be LumbarDisc')
    # 如果Faiss索引文件存在，则直接读取索引文件，否则生成新索引
    logger.info("正在检查索引文件...")
    if os.path.exists(index_file):
        index = faiss.read_index(index_file)
    else:
        index = faiss.IndexFlatL2(feature_vector_length)
        index = faiss.IndexIDMap(index)
    # 连接数据库，创建engine，创建表，并创建绑定session类
    logger.info("正在连接数据库...")
    # 获取目标目录下所有dcm文件的路径
    target_dcm_files = []
    for file in os.listdir(target_dir):
        if os.path.splitext(file)[-1] == '.dcm':
            target_dcm_files.append(os.path.join(target_dir, file))
    # 利用字典key唯一特性，保存SeriesID到内存，同时保存文件相对路径
    descriptions_dict = {}
    savings = {}
    logger.info("开始读取Dicom文件标签...")
    for file in tqdm(target_dcm_files):
        # temp_tags_dict: {tag: value}  need_tags: {tag: description}
        temp_tags_dict = read_specific_tags(file, list(need_tags.keys()))
        descriptions_dict[temp_tags_dict['0020|000e']] = {
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
        savings[f"{temp_tags_dict['0020|000e']}-{temp_tags_dict['0020|0013']}"] = file
    logger.info("正在将信息插入数据库...")
    # 将所有路径对象存入列表，并创建session添加所有对象到表中
    saving_objs = []
    for id, path in savings.items():
        saving_objs.append(DicomFileSavingPath(SeriesSequenceID=id, RelativePath=path))
    with meta_session() as saving_session:
        for saving_obj in saving_objs:
            try:
                saving_session.add(saving_obj)
                saving_session.commit()
            except Exception as e:
                logger.error(f'插入 {saving_obj.SeriesSequenceID} 时出错：{e}')
                saving_session.rollback()
                continue

    # 将所有描述对象存入列表，并创建session添加所有对象到表中, 保存成功的对象添加到列表准备下一步工作
    description_objs = []
    for value in descriptions_dict.values():
        description_objs.append(DescriptionObj(**value))
    description_insert_success = []
    with meta_session() as description_session:
        for description_obj in description_objs:
            try:
                description_session.add(description_obj)
                description_session.commit()
                description_insert_success.append(description_obj)
            except Exception as e:
                logger.error(f'Insert {description_obj.SeriesInstanceUID} failed! Because {e}')
                description_session.rollback()
                continue

        # 装载深度学习模型，为获取特征向量做准备
        if description_insert_success:
            logger.info("有新信息插入，开始载入模型，计算特征向量...")
            model = load_model(tomography_type)
            with meta_session() as query_session:
                # 开始对保存成功的Series信息进行循环，准备计算特征向量
                feature_vectors = []
                index_ids = []
                for obj in tqdm(description_insert_success):
                    # 获取所有的路径表主键
                    primary_keys = [f"{obj.SeriesInstanceUID}-{i}" for i in range(1, int(obj.AcquisitionNumber) + 1)]
                    with TemporaryDirectory() as tmpdir:
                        # 复制一个Series的dicom文件到临时目录
                        for primary_key in primary_keys:
                            file_path = query_session.query(DicomFileSavingPath).get(primary_key).RelativePath
                            shutil.copyfile(file_path, os.path.join(tmpdir, os.path.split(file_path)[-1]))
                        # 开始进行特征提取
                        image_array = read_dicom_dir(tmpdir)
                        feature_vector = get_feature_vector(model, image_array)
                        feature_vectors.append(feature_vector)
                        index_ids.append(int(query_session.query(DescriptionObj).filter(
                            DescriptionObj.SeriesInstanceUID == obj.SeriesInstanceUID).first().IndexID))
                logger.info("正在保存最终文件...")
                features_array = np.concatenate(feature_vectors).astype('float32')
                ids_array = np.array(index_ids).astype('int64')
                index.add_with_ids(features_array, ids_array)
                faiss.write_index(index, index_file)
    logger.success(f'建库流程完成！共插入新数据{len(description_insert_success)}条！')


def rebuild_index_from_database(tomography_type):
    if tomography_type == 'LumbarDisc':
        DescriptionObj = LumbarDiscDescription
        index_file = type_config[tomography_type]['index_file']
        feature_vector_length = type_config[tomography_type]['feature_vector_length']
    else:
        raise ValueError('The file_type parameter must be LumbarDisc')
    logger.info('正在建立索引对象...')
    index = faiss.IndexFlatL2(feature_vector_length)
    index = faiss.IndexIDMap(index)
    logger.info('正在加载深度学习模型...')
    model = load_model(tomography_type)
    logger.info("正在连接数据库...")
    logger.info('开始从数据库重建特征向量索引...')
    with meta_session() as query_session:
        # 开始对所有Series信息进行循环，准备计算特征向量
        all_description_objs = query_session.query(DescriptionObj).all()
        feature_vectors = []
        index_ids = []
        for obj in tqdm(all_description_objs):
            # 获取所有的路径表主键
            primary_keys = [f"{obj.SeriesInstanceUID}-{i}" for i in range(1, int(obj.AcquisitionNumber) + 1)]
            with TemporaryDirectory() as tmpdir:
                # 复制一个Series的dicom文件到临时目录
                for primary_key in primary_keys:
                    file_path = query_session.query(DicomFileSavingPath).get(primary_key).RelativePath
                    shutil.copyfile(file_path, os.path.join(tmpdir, os.path.split(file_path)[-1]))
                # 开始进行特征提取
                image_array = read_dicom_dir(tmpdir)
                feature_vector = get_feature_vector(model, image_array)
                feature_vectors.append(feature_vector)
                index_ids.append(int(query_session.query(DescriptionObj).filter(
                    DescriptionObj.SeriesInstanceUID == obj.SeriesInstanceUID).first().IndexID))
        logger.info("正在保存最终文件...")
        features_array = np.concatenate(feature_vectors).astype('float32')
        ids_array = np.array(index_ids).astype('int64')
        index.add_with_ids(features_array, ids_array)
        faiss.write_index(index, index_file)
        logger.success(f'重建特征向量索引完成！共建立新索引{len(all_description_objs)}条！')


def delete_by_series_id(series_ids, tomography_type):
    if tomography_type == 'LumbarDisc':
        DescriptionObj = LumbarDiscDescription
    else:
        raise ValueError('The file_type parameter must be LumbarDisc')
    with meta_session() as session:
        for series_id in series_ids:
            try:
                target_obj = session.query(DescriptionObj).filter(DescriptionObj.SeriesInstanceUID == series_id).first()
                path_obj_ids = [f"{target_obj.SeriesInstanceUID}-{i}" for i in
                                range(1, int(target_obj.AcquisitionNumber) + 1)]
                for path_obj_id in path_obj_ids:
                    record = session.query(DicomFileSavingPath).get(path_obj_id)
                    session.delete(record)
                session.delete(target_obj)
                session.commit()
            except Exception as e:
                logger.error(f"当删除SeriesID为{series_id}的对象时发生错误: {e}")
                session.rollback()
                continue


def query_by_index_id(index_ids, tomography_type):
    if tomography_type == 'LumbarDisc':
        DescriptionObj = LumbarDiscDescription
    else:
        raise ValueError('The file_type parameter must be LumbarDisc')
    with meta_session() as session:
        results = []
        for index_id in index_ids:
            result = session.query(DescriptionObj).get(int(index_id))
            if result:
                results.append(result)
            else:
                logger.error(f'未查询到Index ID为{index_id}的记录')
        return results


def query_saving_path_by_series_id(series_id):
    with meta_session() as session:
        query_results = session.query(DicomFileSavingPath).filter(
            DicomFileSavingPath.SeriesSequenceID.like(series_id + "%")).all()
        saving_paths = []
        for result in query_results:
            saving_paths.append(result.RelativePath)
        return saving_paths


def search_similar_topn(feature_vector: np.ndarray, top_number: int, tomography_type):
    if tomography_type == 'LumbarDisc':
        index_file = type_config[tomography_type]['index_file']
        feature_vector_length = type_config[tomography_type]['feature_vector_length']
    else:
        raise ValueError('The file_type parameter must be LumbarDisc')
    if not os.path.exists(index_file):
        logger.error('配置指定的Faiss索引文件不存在，无法加载索引文件！')
        return None
    if feature_vector.shape != (1, feature_vector_length):
        logger.error(f'查询{tomography_type}的向量长度不符合要求,应为{(1, feature_vector_length)}')
        return None
    if top_number < 1 or top_number > 20:
        logger.error("寻找相似向量范围不能为负数或大于20！")
        return None
    index = faiss.read_index(index_file)
    search_result = index.search(feature_vector.astype('float32'), top_number)
    search_result_dict = {}
    for i in range(top_number):
        search_result_dict[str(search_result[1][0][i])] = search_result[0][0][i]
    search_result_objs = []
    match_records = query_by_index_id(list(search_result_dict.keys()), tomography_type)
    for record in match_records:
        search_result_objs.append(SearchResult(SeriesRecord=record, Distance=search_result_dict[str(record.IndexID)]))
    return sorted(search_result_objs)


if __name__ == '__main__':
    # build_from_dir('CT_data\LumbarDisc-4Frames', tomography_type='LumbarDisc')
    # ids = [1, 2, 3, 4, 1000, 5]
    # print(query_by_index_id(ids, tomography_type='LumbarDisc')[0])
    rebuild_index_from_database(tomography_type='LumbarDisc')
    # delete_by_series_id(["1.2.156.14702.1.1002.16.1.2018100508584334797563812"], 'LumbarDisc')
    # print(query_saving_path_by_series_id("1.2.156.14702.1.1002.16.1.2018100508584334797563812"))
