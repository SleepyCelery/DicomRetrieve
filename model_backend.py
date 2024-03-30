import numpy as np
import os
import SimpleITK as sitk
from torchvision.transforms import transforms
import torch
import torch.nn as nn
import torchvision.models as models
from config import *

use_cuda = torch.cuda.is_available() and True

data_transforms = transforms.Compose([
    # transforms.Resize([config['image_size'], config['image_size']]), # resize
    # transforms.RandomHorizontalFlip(), # 随机翻转
    transforms.ToTensor(),  # 变成tensor
    transforms.Normalize(
        mean=[-75.97, -75.97, -75.97, -75.97],
        std=[286.35, 286.35, 286.35, 286.35]
    )
])


class Resnet34Triplet(nn.Module):
    def __init__(self, embedding_dimension=128, pretrained=True):
        super(Resnet34Triplet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=3, kernel_size=3, padding=1, stride=2)
        #    self.conv2 = nn.Conv2d(in_channels=4,out_channels=3,kernel_size=3,padding=1,stride=2)
        #    self.conv3 = nn.Conv2d(in_channels=4,out_channels=3,kernel_size=3,padding=1,stride=2)
        #        self.transformer = Transformer(512,10,6,1,1)

        self.model = models.resnet34(pretrained=pretrained)
        input_features_fc_layer = self.model.fc.in_features
        # Output embedding
        self.model.fc = nn.Linear(input_features_fc_layer, embedding_dimension)

    #  self.model.last_fc = nn.Linear(embedding_dimension,264)

    def l2_norm(self, input):
        """Perform l2 normalization operation on an input vector.
        code copied from liorshk's repository: https://github.com/liorshk/facenet_pytorch/blob/master/model.py
        """
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)

        return output

    def forward(self, images):
        """Forward pass to output the embedding vector (feature vector) after l2-normalization and multiplication
        by scalar (alpha)."""
        images = self.conv1(images)
        embedding = self.model(images)
        embedding = self.l2_norm(embedding)
        # Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
        #   Equation 9: number of classes in VGGFace2 dataset = 9131
        #   lower bound on alpha = 5, multiply alpha by 2; alpha = 10
        alpha = 10
        embedding = embedding * alpha

        return embedding


def read_dicom_dir(path, transform: bool = True):
    filenames = os.listdir(path)
    for filename in filenames:
        if os.path.splitext(filename)[-1] != '.dcm':
            filenames.remove(filename)
    reader = sitk.ImageSeriesReader()
    img_names = reader.GetGDCMSeriesFileNames(path)
    reader.SetFileNames(img_names)
    image = reader.Execute()
    image_array = sitk.GetArrayFromImage(image)  # z, y, x
    image_array = image_array.transpose(1, 2, 0).astype('float')
    if transform:
        image_array = data_transforms(image_array)
    if use_cuda:
        image_array = image_array.cuda()
    image_array = image_array.type(torch.float)
    return torch.unsqueeze(image_array, 0)


def load_model(tomography_type):
    if tomography_type == 'LumbarDisc':
        model_file = type_config[tomography_type]['model_file']
        feature_vector_length = type_config[tomography_type]['feature_vector_length']
    else:
        raise ValueError('The file_type parameter must be LumbarDisc')
    model = Resnet34Triplet(pretrained=False, embedding_dimension=int(feature_vector_length))
    model_state = torch.load(model_file)
    model.load_state_dict(model_state['model_state_dict'])
    model.eval()
    if use_cuda:
        model.cuda()
    return model


def get_feature_vector(model, image_array):
    result = model(image_array)
    vector_numpy = result.cpu().detach().numpy()
    return vector_numpy
