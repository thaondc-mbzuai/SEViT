import torch.nn as nn 
import torch
from torchvision import models  # or any other variant of ViT
import timm
class ViTFeatureExtractor(torch.nn.Module):
    def __init__(self, original_vit_model, block_num):
        super().__init__()
        self.vit = original_vit_model
        self.block_num = block_num

    def forward(self, x):
        # Forward pass through the patched embedding
        x = self.vit.patch_embed(x)

        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)  # Expand the cls_token for the batch
        if self.vit.dist_token is not None:
            x = torch.cat((cls_token, self.vit.dist_token, x), dim=1)
        else:
            x = torch.cat((cls_token, x), dim=1)

        x = self.vit.pos_drop(x + self.vit.pos_embed)

        # Forward pass through selected transformer blocks
        for i, blk in enumerate(self.vit.blocks):
            x = blk(x)
            if i == self.block_num:
                break

        return x
    
class Classifier(nn.Module): 
    """
    MLP classifier. 
    Args:
        num_classes -> number of classes 
        in_feature -> features dimension

    return logits. 
    
    """
    def __init__(self,num_classes=2,vit=None, block_num=0):
        super().__init__()
        self.vit = ViTFeatureExtractor(vit, block_num=block_num)
        self.linear1 = nn.Linear(in_features= 196*768, out_features= 4096)
        self.linear2 = nn.Linear(in_features= 4096, out_features= 2048)
        self.linear3 = nn.Linear(in_features= 2048, out_features= 128)
        self.linear4 = nn.Linear(in_features= 128, out_features= num_classes)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self,x):
        x=self.vit(x)
        x=x[:, 1:, :]
        x= x.reshape(-1, 196*768)
        x = nn.functional.relu(self.linear1(x))
        x = nn.functional.relu(self.linear2(x))
        x = nn.functional.relu(self.linear3(x))
        x = self.linear4(x)
        return x

class eficientB0(nn.Module): 
    """
    MLP classifier. 
    Args:
        num_classes -> number of classes 
        in_feature -> features dimension

    return logits. 
    
    """
    def __init__(self, num_classes=2, vit=None, block_num=0):
        super().__init__()
        self.vit = ViTFeatureExtractor(vit, block_num=block_num)
        self.conv1 = nn.Conv2d(in_channels=768, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.eficientB0 = timm.create_model('efficientnet_b0', pretrained=True)
        self.linear = nn.Linear(640, 128) 
        self.classifier = nn.Linear(128, 2) # Not sure
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x=self.vit(x)
        x=x[:,1:,:]
        # print(x.shape)
        x = x.reshape(-1, 768, 14, 14) # reshape patches to image-like format
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.bn1(x)
        # print(x.shape)
        x = self.relu(x)
        # print(x.shape)
        #x = self.maxpool(x)
        # print(x.shape)
        for i in range(3):
            x = self.eficientB0.blocks[i](x) # Not sure
        # print(x.shape)
        #x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1) # x = torch.flatten(x, start_dim=1)
        x = self.dropout(x) # x: (4096)
        x = self.linear(x)
        x = self.dropout(x) 
        #print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x