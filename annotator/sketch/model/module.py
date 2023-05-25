import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1,48,(5, 5),(2, 2),(2, 2)),
            nn.ReLU(),
            nn.Conv2d(48,128,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.Conv2d(128,128,(3, 3),(2, 2),(1, 1)),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(128,256,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.Conv2d(256,256,(3, 3),(2, 2),(1, 1)),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.Conv2d(512,1024,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.Conv2d(1024,1024,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(1024,512,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.Conv2d(512,256,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(256,256,(4, 4),(2, 2),(1, 1),(0, 0)),
            nn.ReLU(),
            nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(256,128,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(128,128,(4, 4),(2, 2),(1, 1),(0, 0)),
            nn.ReLU(),
            nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(128,48,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(48,48,(4, 4),(2, 2),(1, 1),(0, 0)),
            nn.ReLU(),
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(48,24,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(),
            nn.Conv2d(24,1,(3, 3),(1, 1),(1, 1)),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        return x
