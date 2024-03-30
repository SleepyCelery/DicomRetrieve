import SimpleITK as sitk
from description import DicomDictionary


def get_description(group: str, element: str, space: bool = False):
    code = ''.join(['0x', group, element])
    desc = DicomDictionary.get(eval(code))
    if desc:
        if space:
            return desc[2]
        else:
            return desc[4]
    else:
        return ''


def read_tags(path, description: bool = True):
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()
    image = reader.Execute()
    try:
        charcode = image.GetMetaData("0008|0005").strip().encode("utf-8", "surrogateescape").decode()
    except:
        charcode = 'utf-8'
    tags = []
    for i in image.GetMetaDataKeys():
        if "|" in i:
            try:
                if description:
                    tags.append(
                        (i, get_description(*(i.split("|"))),
                         image.GetMetaData(i).strip().encode("utf-8", "surrogateescape").decode(charcode, 'replace')))
                else:
                    tags.append(
                        (
                            i, image.GetMetaData(i).strip().encode("utf-8", "surrogateescape").decode(charcode,
                                                                                                      'replace')))
            except Exception as e:
                if description:
                    tags.append((i, get_description(*i.split("|")), e))
                else:
                    tags.append((i, e))
                continue
    return tags


def read_specific_tag(dcm_file, tag, default=''):
    dcm = sitk.ReadImage(dcm_file)
    try:
        return dcm.GetMetaData(tag).strip().encode("utf-8", "surrogateescape").decode('gbk', 'replace')
    except Exception as e:
        print(e)
        return default


def read_specific_tags(dcm_file, tags: list, default=""):
    dcm = sitk.ReadImage(dcm_file)
    tag_value = {}
    for tag in tags:
        try:
            tag_value[tag] = dcm.GetMetaData(tag).strip().encode("utf-8", "surrogateescape").decode('gbk', 'replace')
        except Exception as e:
            print(e)
            tag_value[tag] = default
    return tag_value


def read_series_in_dir(dir):
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dir)
    return series_ids


if __name__ == '__main__':
    print(read_specific_tag('AllDicomFiles/1.2.156.14702.1.1002.16.2.20181005085952378940.dcm', '0018|1030'))
