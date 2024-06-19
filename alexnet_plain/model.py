import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),  # N x 96 x 55 x 55
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # N x 96 x 27 x 27

            nn.Conv2d(96, 256, kernel_size=5, padding=2),  # N x 256 x 27 x 27
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),  # N x 256 x 13 x 13

            nn.Conv2d(256, 384, kernel_size=3, padding=1),  # N x 384 x 13 x 13
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, padding=1),  # N x 384 x 13 x 13
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # N x 256 x 13 x 13
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)  # N x 256 x 6 x 6
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes)
        )

#         self._initialize_weights()

#     def _initialize_weights(self):
#         for layers in self.modules():
#             if isinstance(layers, nn.Conv2d):
#                 nn.init.normal_(layers.weight, mean=0, std=0.01)
#                 if layers.bias is not None:
#                     nn.init.constant_(layers.bias, 0)

#             elif isinstance(layers, nn.Linear):
#                 nn.init.normal_(layers.weight, mean=0, std=0.01)
#                 if layers.bias is not None:
#                     nn.init.constant_(layers.bias, 1)


    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x