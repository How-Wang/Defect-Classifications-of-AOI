# 新訓 Final Project
- [online link](https://aidea-web.tw/topic/285ef3be-44eb-43dd-85cc-f0388bf85ea4?fbclid=IwAR3NR9UiSB0_u4GxvOfc_xs6b7Bw0dLJfCMnJWpFb5xazd6vzBq5bV7ofDs)
## Paper list
- [TEXTURE IMAGE ANALYSIS AND TEXTURE CLASSIFICATION METHODS - A REVIEW](https://arxiv.org/pdf/1904.06554.pdf)
- [Texture CNN for Thermoelectric Metal Pipe Image Classification](https://www.researchgate.net/publication/339267812_Texture_CNN_for_Thermoelectric_Metal_Pipe_Image_Classification)
- [Deep Learning based Feature Extraction for Texture Classification](https://www.sciencedirect.com/science/article/pii/S1877050920311613)
- [GLCM特徵提取+CNN與MLP模型訓練Covid19分類器](https://www.kaggle.com/code/changshucheng/glcm-cnn-mlp-covid19)
- [Image classification using topological features automatically extracted from graph representation of images](https://www.mlgworkshop.org/2019/papers/MLG2019_paper_7.pdf)
    - Topology studies topological features of spaces: namely, properties preserved under continuous deformations of the space, like the number of connected components, loops, or holes.
## Library
- [MAXPOOL2D](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html)
- [ADAPTIVEMAXPOOL2D](https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveMaxPool2d.html)
- [TORCH.TENSOR.VIEW](https://pytorch.org/docs/stable/generated/torch.Tensor.view.html#torch-tensor-view)
- [TORCH.TENSOR.SIZE](https://pytorch.org/docs/stable/generated/torch.Tensor.size.html#torch-tensor-size)
- [PSPNet code](https://github.com/Lextal/pspnet-pytorch/blob/master/pspnet.py)
- [UPSAMPLE](https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html)
- [CONV2D](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)

## QA
- [Load csv and Image dataset in pytorch](https://stackoverflow.com/questions/65231299/load-csv-and-image-dataset-in-pytorch)
- [See layers' parameters](https://stackoverflow.com/questions/70120715/how-can-i-check-parameters-of-pytorch-networks-layers)
- [RuntimeError: Attempting to deserialize object on a CUDA device](https://stackoverflow.com/questions/56369030/runtimeerror-attempting-to-deserialize-object-on-a-cuda-device)
- [early stopping in PyTorch](https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch)

## Model situation
### 1. PSPNet version
- 0.9561
```python=
class PSPNet(nn.Module):
    def __init__(self, num_classes):
        super(PSPNet, self).__init__()
        
        # Conv layers
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1), # output size (N, 16, 512, 512)
            nn.ReLU(),
            nn.MaxPool2d(2), # output size (N, 16, 256, 256)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1), # output size (N, 32, 256, 256)
            nn.ReLU(),
            nn.MaxPool2d(2), # output size (N, 32, 128, 128)
        )
        # Spatial Pyramid Pooling layers
        self.pool1 = nn.AdaptiveMaxPool2d((1, 1)) # output size (N, 32, 1, 1)
        self.pool2 = nn.AdaptiveMaxPool2d((2, 2)) # output size (N, 32, 2, 2)
        self.pool3 = nn.AdaptiveMaxPool2d((3, 3)) # output size (N, 32, 3, 3)
        self.pool4 = nn.AdaptiveMaxPool2d((6, 6)) # output size (N, 32, 6, 6)
        self.con1 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0) # output size (N, 1, 1, 1)
        self.con2 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0) # output size (N, 1, 2, 2)
        self.con3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0) # output size (N, 1, 3, 3)
        self.con4 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0) # output size (N, 1, 6, 6)
        # Upsampling layers
        self.upsample1 = nn.Upsample(scale_factor=128/1, mode='bilinear', align_corners=True) # output size (N, 1, 128, 128)
        self.upsample2 = nn.Upsample(scale_factor=128/2, mode='bilinear', align_corners=True) # output size (N, 1, 128, 128)
        self.upsample3 = nn.Upsample(scale_factor=128/3, mode='bilinear', align_corners=True) # output size (N, 1, 128, 128)
        self.upsample4 = nn.Upsample(scale_factor=128/6, mode='bilinear', align_corners=True) # output size (N, 1, 128, 128)
        # Conv Classifier layers
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=36, out_channels=18, kernel_size=3, stride=2, padding=1), # output size (N, 18, 64, 64)
            nn.ReLU(),
            nn.MaxPool2d(2), # output size (N, 18, 32, 32)
            nn.Conv2d(in_channels=18, out_channels=9, kernel_size=3, stride=2, padding=1), # output size (N, 9, 16, 16)
            nn.ReLU(),
            nn.MaxPool2d(2), # output size (N, 9, 8, 8)
            nn.Flatten(), # output size (N, 9 * 8 * 8)
            nn.Linear(9 * 8 * 8, 128), # output size (N, 128)
            nn.ReLU(),
            nn.Linear(128, num_classes), # output size (N, 6)
        )
        
    def forward(self, x):
        # CNN layers
        x = self.features(x)
        
        # Spatial Pyramid Pooling
        x1 = self.pool1(x)
        x1 = self.con1(x1) 
        x2 = self.pool2(x)
        x2 = self.con2(x2)
        x3 = self.pool3(x)
        x3 = self.con3(x3)
        x4 = self.pool4(x)
        x4 = self.con4(x4)
        
        # Upsampling
        x1 = self.upsample1(x1)
        x2 = self.upsample2(x2)
        x3 = self.upsample3(x3)
        x4 = self.upsample4(x4)
        
        # Concatenate the pooled features
        x = torch.cat((x1, x2, x3, x4, x), dim=1) # output size (N, 36, 128, 128)
        
        # Classifier
        x = self.classifier(x)
        return x
```
### 2. PSPNet with deeper feature extractor
- score:0.9667077
```python=
class PSPNet(nn.Module):
    def __init__(self, num_classes):
        super(PSPNet, self).__init__()
        
        # Conv layers
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1), # output size (N, 16, 512, 512)
            nn.ReLU(),
            nn.MaxPool2d(2), # output size (N, 16, 256, 256)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1), # output size (N, 32, 256, 256)
            nn.ReLU(),
            nn.MaxPool2d(2), # output size (N, 32, 128, 128)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1), # output size (N, 64, 128, 128)
            nn.ReLU(),
            nn.MaxPool2d(2), # output size (N, 64, 64, 64)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), # output size (N, 128, 64, 64)
            nn.ReLU(),
            nn.MaxPool2d(2), # output size (N, 128, 32, 32)
        )
        # Spatial Pyramid Pooling layers
        self.pool1 = nn.AdaptiveMaxPool2d((1, 1)) # output size (N, 64, 1, 1)
        self.pool2 = nn.AdaptiveMaxPool2d((2, 2)) # output size (N, 64, 2, 2)
        self.pool3 = nn.AdaptiveMaxPool2d((3, 3)) # output size (N, 64, 3, 3)
        self.pool4 = nn.AdaptiveMaxPool2d((6, 6)) # output size (N, 64, 6, 6)
        self.con1 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0) # output size (N, 1, 1, 1)
        self.con2 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0) # output size (N, 1, 2, 2)
        self.con3 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0) # output size (N, 1, 3, 3)
        self.con4 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0) # output size (N, 1, 6, 6)
        # Upsampling layers
        self.upsample1 = nn.Upsample(scale_factor=32/1, mode='bilinear', align_corners=True) # output size (N, 1, 32, 32)
        self.upsample2 = nn.Upsample(scale_factor=32/2, mode='bilinear', align_corners=True) # output size (N, 1, 32, 32)
        self.upsample3 = nn.Upsample(scale_factor=32/3, mode='bilinear', align_corners=True) # output size (N, 1, 32, 32)
        self.upsample4 = nn.Upsample(scale_factor=32/6, mode='bilinear', align_corners=True) # output size (N, 1, 32, 32)
        # Conv Classifier layers
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=132, out_channels=64, kernel_size=3, stride=2, padding=1), # output size (N, 64, 16, 16)
            nn.ReLU(),
            nn.MaxPool2d(2), # output size (N, 64, 8, 8)
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1), # output size (N, 32, 4, 4)
            nn.ReLU(),
            nn.MaxPool2d(2), # output size (N, 32, 2, 2)
            nn.Flatten(), # output size (N, 32 * 2* 2)
            nn.Linear(32 * 2 * 2, 32), # output size (N, 32)
            nn.ReLU(),
            nn.Linear(32, num_classes), # output size (N, 6)
        )
        
    def forward(self, x):
        # CNN layers
        x = self.features(x)
        
        # Spatial Pyramid Pooling
        x1 = self.pool1(x)
        x1 = self.con1(x1) 
        x2 = self.pool2(x)
        x2 = self.con2(x2)
        x3 = self.pool3(x)
        x3 = self.con3(x3)
        x4 = self.pool4(x)
        x4 = self.con4(x4)
        
        # Upsampling
        x1 = self.upsample1(x1)
        x2 = self.upsample2(x2)
        x3 = self.upsample3(x3)
        x4 = self.upsample4(x4)
        
        # Concatenate the pooled features
        x = torch.cat((x1, x2, x3, x4, x), dim=1) # output size (N, 132, 32, 32)
        
        # Classifier
        x = self.classifier(x)
        return x
```


### 3. PSPNet with deeper feature extractor with horizontal and vertical flip
- score:0.9802712
### 4. Base with deeper
- score:0.9701602
```python=
class BasicCNN(nn.Module):
    def __init__(self, num_classes):
        super(BasicCNN, self).__init__()
        
        # Conv layers
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1), # output size (N, 16, 512, 512)
            nn.ReLU(),
            nn.MaxPool2d(2), # output size (N, 16, 256, 256)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1), # output size (N, 32, 256, 256)
            nn.ReLU(),
            nn.MaxPool2d(2), # output size (N, 32, 128, 128)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1), # output size (N, 64, 128, 128)
            nn.ReLU(),
            nn.MaxPool2d(2), # output size (N, 64, 64, 64)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), # output size (N, 128, 64, 64)
            nn.ReLU(),
            nn.MaxPool2d(2), # output size (N, 128, 32, 32)
        )
        # Conv Classifier layers
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1), # output size (N, 64, 16, 16)
            nn.ReLU(),
            nn.MaxPool2d(2), # output size (N, 64, 8, 8)
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1), # output size (N, 32, 4, 4)
            nn.ReLU(),
            nn.MaxPool2d(2), # output size (N, 32, 2, 2)
            nn.Flatten(), # output size (N, 32 * 2* 2)
            nn.Linear(32 * 2 * 2, 32), # output size (N, 32)
            nn.ReLU(),
            nn.Linear(32, num_classes), # output size (N, 6)
        )
        
    def forward(self, x):
        # CNN layers
        x = self.features(x)
        x = self.classifier(x)
        return x
```