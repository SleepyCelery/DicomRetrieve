# CT图像检索服务器端开发文档

## 数据库

对于整个服务而言，需要两个表

表1：Descriptions 存放dcm文件描述数据，使用序列编号作为主键，具体需要存放的字段包括：

1. 主键：'0020|000e', 'SeriesInstanceUID', Required
2. IndexID：Faiss对应向量存储ID(Unique, AutoIncrement)
3. '0010|0010', 'PatientName', Required, Empty if Unknown
4. '0010|0040', 'PatientSex', Required, Empty if Unknown
5. '0010|0030', 'PatientBirthDate', Required, Empty if Unknown
6. '0010|1010', 'PatientAge', Optional
7. '0020|0012', 'AcquisitionNumber', Required, Empty if Unknown
9. '0018|1030', 'ProtocolName', Optional
10. '0008|0020', 'StudyDate', Required, Empty if Unknown
11. '0008|0030', 'StudyTime', Required, Empty if Unknown
12. '0008|0080', 'InstitutionName', Optional

表2：DicomFiles 存放dcm文件位置

1. 主键：SeriesInstanceUID-InstanceNumber(0020,0013)(range from 1 to SeriesDicomFilesCount)
2. RelativePath

## 离线构建过程

1. 读取目标目录下的所有dcm文件，并提取其需要的tags，拼接文件的相对路径，并读取或创建新的Faiss Index（判断是否存在索引文件）
2. 构建描述表和路径表对应的数据行，添加到数据库并提交，并将成功存入数据库的所有数据对象保存到列表中
3. 添加完目录下的所有文件后，对列表中的SeriesID进行迭代，读取其AcquisitionNumber和SeriesID以及自增的IndexID
4. 拼接所有的dcm文件路径表主键，读取目录信息，然后移动到临时目录，此时临时目录内都是同一个Series的Dicom文件
5. 对目录下所有Dicom文件进行读取、图像预处理，并使用模型推理出特征向量
6. 向Faiss索引中添加带id的记录，id为数据库内自增的IndexID
7. 保存Faiss索引
