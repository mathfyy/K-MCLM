import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import pdb
import numpy as np

data_path = '/data/students/wyj/chestmnist/pretrain_model/'

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=4):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.maxpool2 = nn.MaxPool2d(7, 7)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.features = nn.Sequential(self.conv1,
        #                               self.bn1,
        #                               self.relu,
        #                               self.maxpool,
        #                               self.layer1,
        #                               self.layer2,
        #                               self.layer3,
        #                               self.layer4
        #                            )

        # self.layer5 = nn.Conv2d(2048, num_classes*4, kernel_size=2, stride=1)
        # self.layer6 = nn.Conv2d(num_classes*4, num_classes, kernel_size=1, stride=1)
        # self.classifier = nn.Sequential(
        #    nn.Conv2d(2048, 256, kernel_size=2, stride=1))

        # self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # self.fc1 = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print(x.shape)
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # embeddings = self.maxpool2(x4)  # 8 * 2048 * 1 * 1
        # embeddings = embeddings.view(embeddings.size(0), -1)  # 8 * 2048
        # logits = self.fc1(x)

        outputs = {
            # "embeddings": embeddings,
            "attentions": [x1, x2, x3, x4]
        }
        return outputs


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # self.device = device
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # print('GCN is running')
        # print(input.shape)
        # input = torch.Tensor(input)
        # input = input.to(self.device)
        support = torch.matmul(input, self.weight)
        # support = support.to(self.device)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'

class labelCorModule_s(nn.Module):
    def __init__(self, ):
        super(labelCorModule_s, self).__init__()
        self.tanh = nn.Tanh()
        self.upCor_s = upCor_s()
        self.downCor_s = downCor_s()

    def forward(self, Ag, V0, M, Ag_param):
        # x0 = Ag

        cor_D = torch.bmm(V0, V0.permute(0, 2, 1)) / V0.shape[2]
        cor_P = torch.bmm(M.permute(0, 2, 1), M) / M.shape[1]
        x0 = Ag + torch.mean(
            Ag_param[0, 0] * (self.tanh(cor_D.unsqueeze(1)).squeeze(1)) + Ag_param[0, 1] * (
                self.tanh(cor_P.unsqueeze(1)).squeeze(1)), dim=0)

        x0 = x0.unsqueeze(0).unsqueeze(0)

        # 0
        # out = x0

        # 1
        # P, Q, Atte = self.upCor_s(x0)
        # out = self.downCor_s(x0, P, Q)
        # out = Ag_param[0, 2] * out + x0

        # 2
        P, Q, Atte = self.upCor_s(x0, True)
        P1, Q1, Atte1 = self.upCor_s(Atte.unsqueeze(1))
        out = self.downCor_s(x0, P, self.downCor_s(Q.unsqueeze(1), P1, Q1))
        out = Ag_param[0, 2] * out + x0

        # 3
        # P, Q, Atte = self.upCor_s(x0, True)
        # P1, Q1, Atte1 = self.upCor_s(Atte.unsqueeze(1), True)
        # P2, Q2, Atte2 = self.upCor_s(Atte1.unsqueeze(1))
        # dQ2 = self.downCor_s(Q1.unsqueeze(1), P2, Q2)
        # dQ1=self.downCor_s(Q.unsqueeze(1), P1, dQ2)
        # out = self.downCor_s(x0, P, dQ1)
        # out = Ag_param[0, 2] * out + x0

        return out


class upCor_s(nn.Module):
    def __init__(self, ):
        super(upCor_s, self).__init__()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        in_dim = 1
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

    def forward(self, x, is_atte=False):
        m_batchsize, C, height, width = x.size()
        P = self.sigmoid(self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1))
        Q = self.sigmoid(self.key_conv(x).view(m_batchsize, -1, width * height))
        Atte = []
        if is_atte is True:
            Atte = torch.bmm(P, Q) / (width * height)
        return P, Q, Atte


class downCor_s(nn.Module):
    def __init__(self, ):
        super(downCor_s, self).__init__()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, P, Q):
        m_batchsize, C, height, width = x.size()
        x = x.view(m_batchsize, -1, width * height)
        P = P.view(m_batchsize, width * height, -1)
        Q = Q.view(m_batchsize, -1, width * height)
        out1 = torch.bmm(x, P) / (width * height)
        out1 = self.tanh(out1)
        out = torch.bmm(out1, Q) / (width * height)
        out = out.view(m_batchsize, height, width)
        out = self.tanh(out)
        return out


class cadBlock(nn.Module):
    def __init__(self, device, mid_dim=2048, head_dim=512):
        super(cadBlock, self).__init__()
        self.device = device
        self.mid_dim = mid_dim
        self.device = device
        self.mlp = nn.Linear(mid_dim, 1)

    def forward(self, F_data, A_cls):
        F_data = F_data.reshape(F_data.shape[0], F_data.shape[1], F_data.shape[2] * F_data.shape[3]).transpose(2, 1)

        scores = F.softmax(torch.matmul(A_cls.repeat(F_data.shape[0], 1, 1), F_data.transpose(-2, -1))/F_data.shape[1], dim=-1)
        Hba = torch.matmul(scores, F_data)

        scores = scores.permute(0, 2, 1)
        out = self.mlp(Hba).reshape(Hba.shape[0], Hba.shape[1])

        return scores, Hba, out


class cscBlock(nn.Module):
    def __init__(self, device, D0, D1, D2, mid_dim=2048, head_dim=512):
        super(cscBlock, self).__init__()
        self.device = device
        self.D0 = D0
        self.D1 = D1
        self.D2 = D2
        self.mid_dim = mid_dim
        self.head_dim = head_dim
        # self.gp = nn.AdaptiveAvgPool2d((1, 1))

        self.cad = cadBlock(device)

        self.generalGCN = GraphConvolution(D0, D1)
        self.lRelu = nn.LeakyReLU(inplace=True)

        self.sigmoid = nn.Sigmoid()

        self.labelCorModule = labelCorModule_s()

        self.fc = nn.Linear(D2, 1)

    def forward(self, class_token, F, Ag_param):
        Ag = torch.tanh(torch.mm(class_token, class_token.permute(1, 0)) / class_token.shape[1])

        scores, V0, output1 = self.cad(F, class_token)

        Ag_ = self.labelCorModule(Ag, V0, scores, Ag_param)

        V1 = torch.ones(size=(V0.size(0), V0.size(1), self.D1)).to(self.device)
        for i in range(V0.size(0)):
            V1[i,] = self.lRelu(self.generalGCN(V0[i, :, :], Ag_))

        output2 = self.fc(V1).squeeze(dim=2)
        output = self.sigmoid(output1 + output2)

        return output


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        '''
        pretrained_dict = model_zoo.load_url(model_urls['resnet101'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        '''
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet101']),strict=False)
        # cocodata
        # model.load_state_dict(torch.load(data_path + '/resnet101-5d3b4d8f.pth'), strict=False)
        # chestdata
        model.load_state_dict(torch.load(data_path + '/resnext101_64x4d-e77a0586.pth'), strict=False)

    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
