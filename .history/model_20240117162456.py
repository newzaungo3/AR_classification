from torch import nn

class ResidualBlock(nn.Module):
    
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride = 1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv3 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels*self.expansion, kernel_size = 1, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels*self.expansion),
                        nn.ReLU())
        self.downsample = None
        if stride != 1 or in_channels != out_channels * 4:
            print("HELLO")
            self.downsample = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                      out_channels=out_channels*self.expansion,
                                                      kernel_size=1,
                                                      stride=stride),
                                            nn.BatchNorm2d(out_channels*self.expansion))
        
        # self.conv2 = nn.Sequential(
        #                 nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
        #                 nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        
        identity = x.clone()
        print(identity.shape)
        out = self.conv1(x)
        print(out.shape)
        out = self.conv2(out)
        print(out.shape)
        out = self.conv3(out)
        print(out.shape)
        if self.downsample:
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out
    
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 3):
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, 64 ,layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 64 * 4,128 ,layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 128 * 4,256,layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 256 * 4,512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(7, stride=1)    
        self.fc = nn.Linear(512 * 4, num_classes)
        
    def _make_layer(self, block,in_channels,out_channels, blocks, stride=1):
        """
        Create a layer with specified type and number of residual blocks.
        Args: 
            ResBlock: residual block type, BasicBlock for ResNet-18, 34 or 
                      BottleNeck for ResNet-50, 101, 152
            n_blocks: number of residual blocks
            in_channels: number of input channels
            out_channels: number of output channels
            stride: stride used in the first 3x3 convolution of the first resdiual block
            of the layer and 1x1 convolution for skip connection in that block
        Returns: 
            Convolutional layer
        """
        layer = []
        for i in range(blocks):
            if i == 0:
                # Downsample the feature map using input stride for the first block of the layer.
                layer.append(block(in_channels, out_channels, 
                             stride=stride))
            else:
                # Keep the feature map size same for the rest three blocks of the layer.
                # by setting stride=1 and is_first_block=False.
                # By default, ResBlock.expansion = 4 for ResNet-50, 101, 152, 
                # ResBlock.expansion = 1 for ResNet-18, 34.
                layer.append(block(out_channels*block.expansion, out_channels))

        return nn.Sequential(*layer)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        print(x.shape)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x