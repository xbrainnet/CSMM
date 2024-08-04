import torch.nn as nn
from functions import ReverseLayerF


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.view((x.size(0),)+self.shape)




class SANN(nn.Module):

    def __init__(self):
        super(SANN, self).__init__()
        self.shallow = nn.Sequential()
        self.shallow.add_module('s_fc1', nn.Linear(310, 256))
        self.shallow.add_module('s_bn1', nn.BatchNorm1d(256))
        self.shallow.add_module('s_relu1', nn.ReLU())
        self.shallow.add_module('s_drop1', nn.Dropout())
        self.shallow.add_module('s_fc2', nn.Linear(256, 128))
        self.shallow.add_module('s_bn2', nn.BatchNorm1d(128))
        self.shallow.add_module('s_relu2', nn.ReLU())


        self.deep = nn.Sequential()
        self.deep.add_module('de_fc1', nn.Linear(128, 32))
        self.deep.add_module('de_relu1', nn.ReLU())
        # self.deep.add_module('de_fc2', nn.Linear(64, 32))
        # self.deep.add_module('de_relu2', nn.ReLU())


        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(32, 4))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))


        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(128, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))


    def forward(self, input_data, alpha):
        feature = self.shallow(input_data)
        feature_deep = self.deep(feature)
        class_output = self.class_classifier(feature_deep)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        domain_output = self.domain_classifier(reverse_feature)
        return class_output, domain_output


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight)
        nn.init.constant_(m.bias, 0.1)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


