import torch.nn              as nn
import torchvision.models    as models
import torch.utils.model_zoo as model_zoo


class Model(nn.Module):

    def freeze():
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze():
        for param in self.parameters():
            param.requires_grad = True




class AlexNet(torchvision.models.AlexNet):

    model.load_state_dict(model_zoo.load_url(settings['url']))
    model.input_space = "RGB"
    model.input_size  = [3, 224, 224]
    model.input_range = [0, 1]
    model.mean        = [0.485, 0.456, 0.406]
    model.std         = [0.229, 0.224, 0.225]

class VisionModel(Model):

    def __init__(self, model_name, pretrained=True):
        # super(TwoLayerNet, self).__init__()
        # super().__init__(x_transform, y_transform)

        try:
            model_fn = eval("torchvision.models."+model_name) # Create model fn
            self     = model_fn(pretrained=pretrained)        # Create model

        except AttributeError:
            print(model_name+" model don't exists in torchvision")


    def change_last_layer(num_outputs):
        self.freeze()
        num_features  = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_outputs) # New last layer is unfreezed


# see https://github.com/Cadene/pretrained-models.pytorch


model.input_size

    [3, 299, 299] for inception* networks,
    [3, 224, 224] for resnet* networks.

model.input_space

    RGB or BGR

model.input_range

    [0, 1] for resnet* and inception* networks,
    [0, 255] for bninception network.


model.mean

    [0.5, 0.5, 0.5] for inception* networks,
    [0.485, 0.456, 0.406] for resnet* networks.

model.std

    [0.5, 0.5, 0.5] for inception* networks,
    [0.229, 0.224, 0.225] for resnet* networks.

model_names = [
    'alexnet',
    'densenet121', 'densenet169', 'densenet201', 'densenet161',
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'inceptionv3',
    'squeezenet1_0', 'squeezenet1_1',
    'vgg11',    'vgg13',    'vgg16',    'vgg19',
    'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn',
]

pretarined = {
    "alexnet":       "https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth",
    "densenet121":   "https://download.pytorch.org/models/densenet121-241335ed.pth",
    "densenet169":   "https://download.pytorch.org/models/densenet169-6f0f7f60.pth",
    "densenet201":   "https://download.pytorch.org/models/densenet201-4c113574.pth",
    "densenet161":   "https://download.pytorch.org/models/densenet161-17b70270.pth",   
    "inceptionv3":   "https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth",
    "resnet18":      "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34":      "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50":      "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101":     "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152":     "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
    "squeezenet1_0": "https://download.pytorch.org/models/squeezenet1_0-a815701f.pth",
    "squeezenet1_1": "https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth",
    "vgg11":         "https://download.pytorch.org/models/vgg11-bbd30ac9.pth",
    "vgg13":         "https://download.pytorch.org/models/vgg13-c768596a.pth",
    "vgg16":         "https://download.pytorch.org/models/vgg16-397923af.pth",
    "vgg19":         "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
    "vgg11_bn":      "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
    "vgg13_bn":      "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
    "vgg16_bn":      "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
    "vgg19_bn":      "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
    # "vgg16_caffe":   "https://s3-us-west-2.amazonaws.com/jcjohns-models/vgg16-00b39a1b.pth",
    # "vgg19_caffe":   "https://s3-us-west-2.amazonaws.com/jcjohns-models/vgg19-d01eb7cb.pth"
}

# resnet* networks
for model_name in model_names:
    input_sizes[model_name] = [3, 224, 224]
    means[model_name]       = [0.485, 0.456, 0.406]
    stds[model_name]        = [0.229, 0.224, 0.225]

# inception* networks
for model_name in ['inceptionv3']:
    input_sizes[model_name] = [3, 299, 299]
    means[model_name]       = [0.5, 0.5, 0.5]
    stds[model_name]        = [0.5, 0.5, 0.5]


for model_name in model_names:
    pretrained_settings[model_name] = {
        'imagenet': {
            'url':         pretarined[model_name],
            'input_space': 'RGB',
            'input_size':  input_sizes[model_name],
            'input_range': [0, 1],
            'mean':        means[model_name],
            'std':         stds[model_name],
            'num_classes': 1000
        }
    }

