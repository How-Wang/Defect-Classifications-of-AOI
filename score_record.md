# 新訓 Final Project
- [online link](https://aidea-web.tw/topic/285ef3be-44eb-43dd-85cc-f0388bf85ea4?fbclid=IwAR3NR9UiSB0_u4GxvOfc_xs6b7Bw0dLJfCMnJWpFb5xazd6vzBq5bV7ofDs)
## Paper list
- ==[Pyramid Scene Parsing Network](https://arxiv.org/pdf/1612.01105.pdf)==
    - ![](https://hackmd.io/_uploads/rkJV2wx02.png)
- ==[Medical Image Classification Based on Deep Features Extracted by Deep Model and Statistic Feature Fusion with Multilayer Perceptron](https://www.hindawi.com/journals/cin/2018/2061516/)==
    - ![](https://hackmd.io/_uploads/SysUnwg0h.png)
- [TEXTURE IMAGE ANALYSIS AND TEXTURE CLASSIFICA
TION METHODS - A REVIEW](https://arxiv.org/pdf/1904.06554.pdf)
- [Texture CNN for Thermoelectric Metal Pipe Image Classification](https://www.researchgate.net/publication/339267812_Texture_CNN_for_Thermoelectric_Metal_Pipe_Image_Classification)
- [Deep Learning based Feature Extraction for Texture Classification](https://www.sciencedirect.com/science/article/pii/S1877050920311613)
- [GLCM特徵提取+CNN與MLP模型訓練Covid19分類器](https://www.kaggle.com/code/changshucheng/glcm-cnn-mlp-covid19)
- [Image classification using topological features automatically extracted from graph representation of images](https://www.mlgworkshop.org/2019/papers/MLG2019_paper_7.pdf)
    - Topology studies topological features of spaces: namely, properties preserved under continuous deformations of the space, like the number of connected components, loops, or holes.
- [應用多重特徵於提升材質辨識的準確性](https://ir.nctu.edu.tw/bitstream/11536/73614/1/007701.pdf)
    - 由上述情況發現GLCM+LBP多重特徵除了在光亮環境變化下能提升其強健性，而其他環境變化提升幅度有限。其中尺度變化發生上述問題不能具有代表性。造成上述狀況推測GLCM與LBP特性過於相似，所以當其中一種特徵抽取方式無法抵抗環境變化的問題，而另一特徵抽取方式也無法抵抗相同環境變化問題，造成多重特徵無法提升整體強健性。
 
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
- [大佬，报错TypeError: unsupported operand type(s) for /: 'PngImageFile' and 'int'](https://github.com/nickliqian/cnn_captcha/issues/82)
## To-do
- [x] dataset blur
- [x] check GLCM features
## Model situation
- 
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
        self.pool1 = nn.AdaptiveMaxPool2d((1, 1)) # output size (N, 128, 1, 1)
        self.pool2 = nn.AdaptiveMaxPool2d((2, 2)) # output size (N, 128, 2, 2)
        self.pool3 = nn.AdaptiveMaxPool2d((3, 3)) # output size (N, 128, 3, 3)
        self.pool4 = nn.AdaptiveMaxPool2d((6, 6)) # output size (N, 128, 6, 6)
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
### 4. Base with deeper feature extractor with horizontal and vertical flip
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
### 5. PSPNet with deeper feature extractor with horizontal and vertical flip and GLCM feature
- score:0.9815043
```python=
def GLCM_features(image):
    image = np.array(image)
    image = (image * 255).astype(np.uint8)
    glcm_features = torch.empty(25, dtype=torch.float32)

    #5 configuration for the grey-level co-occurrence matrix calculation
    dists = [[1],[3],[5],[3],[3]]
    angles = [[0],[0],[0],[np.pi/4],[np.pi/2]]

    for j ,(dist, angle) in enumerate(zip(dists, angles)):
        GLCM = graycomatrix(image, dist, angle) 
        glcm_features[j*5] = torch.tensor(graycoprops(GLCM, 'energy')[0], dtype=torch.float32)
        glcm_features[j*5 + 1] = torch.tensor(graycoprops(GLCM, 'correlation')[0] , dtype=torch.float32)   
        glcm_features[j*5 + 2] = torch.tensor(graycoprops(GLCM, 'dissimilarity')[0], dtype=torch.float32)
        glcm_features[j*5 + 3] = torch.tensor(graycoprops(GLCM, 'homogeneity')[0], dtype=torch.float32)
        glcm_features[j*5 + 4] = torch.tensor(graycoprops(GLCM, 'contrast')[0], dtype=torch.float32)
        
    return glcm_features
```
```python=
class PSPNetGLCM(nn.Module):
    def __init__(self, num_classes):
        super(PSPNetGLCM, self).__init__()
        
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
        self.pool1 = nn.AdaptiveMaxPool2d((1, 1)) # output size (N, 128, 1, 1)
        self.pool2 = nn.AdaptiveMaxPool2d((2, 2)) # output size (N, 128, 2, 2)
        self.pool3 = nn.AdaptiveMaxPool2d((3, 3)) # output size (N, 128, 3, 3)
        self.pool4 = nn.AdaptiveMaxPool2d((6, 6)) # output size (N, 128, 6, 6)
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
        self.nn_classifier = nn.Sequential(
            nn.Conv2d(in_channels=132, out_channels=64, kernel_size=3, stride=2, padding=1), # output size (N, 64, 16, 16)
            nn.ReLU(),
            nn.MaxPool2d(2), # output size (N, 64, 8, 8)
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1), # output size (N, 32, 4, 4)
            nn.ReLU(),
            nn.MaxPool2d(2), # output size (N, 32, 2, 2)
            nn.Flatten(), # output size (N, 32 * 2* 2)
            nn.Linear(32 * 2 * 2, 24), # output size (N, 24)
            nn.ReLU(),
        )
        self.glcm_classifier= nn.Sequential(
            nn.Linear(25, 8), # output size (N, 8)
            nn.ReLU(),
        )
        self.final_classifier = nn.Sequential(
            nn.Linear(32, num_classes) # output size (N, num_classes=6)
        )
        
    def forward(self, x_input, x_glcm):
        # CNN layers
        x = self.features(x_input)
        
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
        x = self.nn_classifier(x) # output size (N, 24)
        
        # Get GLCM features 
        x_glcm = self.glcm_classifier(x_glcm) # output size (N, 8)
        
        # Concatenate nn features and GLCM features
        x = torch.cat((x, x_glcm), dim=1) # output size (N, 32, 32)
        
        # final classifier
        x = self.final_classifier(x)
        
        return x
```