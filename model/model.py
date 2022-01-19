import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from model import ResNet3d


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def general_model(model_type, num_classes):
    smp_size = 16 * 5
    smp_dur = 64

    if model_type == "ResNet10":
        model = ResNet3d.ResNet(ResNet3d.BasicBlock, [1, 1, 1, 1], sample_size=smp_size,
                                  sample_duration=smp_dur, num_classes=num_classes)

    if model_type == 'ResNet34':
        model = ResNet3d.ResNet(ResNet3d.BasicBlock, [3, 4, 6, 3], sample_size=smp_size,
                                  sample_duration=smp_dur, num_classes=num_classes)

    return model