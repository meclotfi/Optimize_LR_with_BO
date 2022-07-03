import torch
import torch.nn as nn
import torch.nn.functional as F



class Resblock(nn.Module):
  """
  Residual network block with the following structure :
                - Conv2D
                - Batch Normalisation
                - Relu
                - Conv2D
                - Batch Normalisation
                - Residual Connection
  
  """
  def __init__(self,in_channels,out_channels):
        super(Resblock, self).__init__()

        self.cnn1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.cnn2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(out_channels),
            )
  def forward(self,x):
    out =F.relu(self.cnn1(x))
    out = self.norm1(out)
    out = self.cnn2(out)
    return self.norm2(out)+self.downsample(x)

class TinyResnet(nn.Module):
    """ TinyResnet is a simple resnet model for classification
    
        Model Structure:
            2x Resblock Blocks:
                - Conv2D
                - Batch Normalisation
                - Relu
                - Conv2D
                - Batch Normalisation
                - Residual Connection
      
            1x Average Pooling Layer
      
            1x Fully Connected Layer:
                - Output Layer
    """

    def __init__(self,num_layers=2,channels=[1,64,128]):
        super(TinyResnet, self).__init__()
        self.blocks=nn.ModuleList()

        for i in range(num_layers):
          self.blocks.append(Resblock(in_channels=channels[i],out_channels=channels[i+1]))
          norm = nn.BatchNorm2d(channels[i+1])
          self.blocks.append(norm)

        self.avg_pool=nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc=nn.Linear(in_features=channels[-1], out_features=10, bias=True)

    def forward(self, x):
        out=x
        for block in self.blocks:
          out=block(out)
        out=self.avg_pool(out)
        out = torch.flatten(out, 1)
        out=self.fc(out)
        return out