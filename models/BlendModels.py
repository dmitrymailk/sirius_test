from torch.nn.parameter import Parameter
import torch 
from torch import nn

class BlendModels(nn.Module):
    def __init__(self, n_classes=10, models=None):
        super(BlendModels, self).__init__()
        assert len(models) != 0, "Expected at least one model"

        for i in range(len(models)):
            for param in models[i].parameters():
                param.requires_grad = False
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_classes = n_classes
        self.models = models
        self.blender_weights = Parameter(torch.ones((len(models))))

    def forward(self, x):
        temp = Parameter(torch.zeros((x.shape[0], self.n_classes), device=self.device))

        for i in range(len(self.models)):
            temp = temp + self.blender_weights[i] * self.models[i](x)
        temp /= len(self.models)

        return temp 