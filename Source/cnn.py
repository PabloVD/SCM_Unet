from torch import nn

def miniblock(in_channels, out_channels, stride=2, kernel_size=3):
    block = nn.Sequential(
                nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, stride=stride),# padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
                #nn.MaxPool2d(kernel_size=2)
            )
    return block

class HmapNet(nn.Module):

    def __init__(self, mid_channels=64):
        super().__init__()

        layers = [
                    miniblock(1, mid_channels//2),
                    miniblock(mid_channels//2, mid_channels),
                    miniblock(mid_channels, mid_channels//2)
                 ]
        
        self.layers = nn.Sequential(*layers)
        self.finalpool = nn.MaxPool2d(kernel_size=[8,4])
        
    def forward(self, x):

        x = self.layers(x)
        x = self.finalpool(x)
        
        return  x