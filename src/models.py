import torch.nn as nn
import torch
from efficientnet_pytorch import EfficientNet

class EfficientNetB5(nn.Module):

    def __init__(self, pretrained=True):
        # init superclass
        super(EfficientNetB5, self).__init__()
        # build model
        self.pretrained = pretrained
        self.__build_model()

    def __build_model(self):
        """
        Model Components
        """
        if self.pretrained: self.backbone_model = EfficientNet.from_pretrained('efficientnet-b5')
        else: self.backbone_model = EfficientNet.from_name('efficientnet-b5')

        # All layers except last FC (_fc) and Swish layers
        # backbone_layers = torch.nn.ModuleList(backbone_model.children())[:-2]
        # self.features = torch.nn.Sequential(*backbone_layers)
        # in_features = self.backbone_model._fc.in_features

        in_features = self.backbone_model._fc.out_features # out_feat. should be an apt name
        # Add custom layers to the model
        self.fc_grapheme_root = torch.nn.Linear(in_features, 168)
        self.fc_vowel_diacritic = torch.nn.Linear(in_features, 11)
        self.fc_consonant_diacritic = torch.nn.Linear(in_features, 7)

    def forward(self, x):
        x = self.backbone_model(x)
        x = torch.flatten(x, 1)
        
        grapheme = self.fc_grapheme_root(x)
        vowel = self.fc_vowel_diacritic(x)
        consonant = self.fc_consonant_diacritic(x) 

        return grapheme, vowel, consonant
