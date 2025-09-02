"""
@author: Hasnat Md Abdullah
@date: Aug 28, 2025
"""

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class JEPA_Encoder(nn.Module):
    def __init__(self, proj_dim=256, imgformat='rgb', pretrained=True):
        super().__init__()
        if pretrained:
            base = models.resnet18(weights='ResNet18_Weights.DEFAULT') # no pretraining
            print("Using ImageNet pretrained ResNet18 as backbone")
        
        else:
            base = models.resnet18(weights = None)
            print("Training ResNet18 from scratch")

        # Modify first conv if grayscale
        if imgformat == 'gray':
            conv1 = base.conv1
            base.conv1 = nn.Conv2d(
                in_channels=1,
                out_channels=conv1.out_channels,
                kernel_size=conv1.kernel_size,
                stride=conv1.stride,
                padding=conv1.padding,
                bias=conv1.bias is not None
            )
            with torch.no_grad():
                base.conv1.weight[:,0] = conv1.weight[:,0]
        
        # Remove classification head       
        self.backbone = nn.Sequential(*list(base.children())[:-2])  # [B,512,H/32,W/32]
                
        # projection head to reduce dimensionality
        self.proj_head = nn.Sequential(
            nn.Conv2d(512, proj_dim, kernel_size=1),
            nn.BatchNorm2d(proj_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        feat = self.backbone(x)          # [B,512,H/32,W/32]
        proj_feat = self.proj_head(feat) # [B,proj_dim,H/32,W/32]
        return proj_feat

class JEPA_Predictor(nn.Module):
    def __init__(self, channels=256):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.predictor(x)
    

class JEPA_FeatureMask(nn.Module):
    def __init__(self, context_encoder, target_encoder, predictor, grid_size=7, context_ratio=0.5):
        super().__init__()
        self.context_encoder = context_encoder
        self.target_encoder = target_encoder
        self.predictor = predictor
        self.grid_size = grid_size
        self.context_ratio = context_ratio

    def create_feature_masks(self, batch_size):
        """
        Creates boolean masks over the 7x7 feature grid
        context_mask = visible tokens
        target_mask = masked tokens
        """
        num_tokens = self.grid_size * self.grid_size
        num_context = int(num_tokens * self.context_ratio)
        context_masks = []
        target_masks = []
        for _ in range(batch_size):
            idx = torch.randperm(num_tokens)
            context_idx = idx[:num_context]
            target_idx = idx[num_context:]
            cm = torch.zeros(num_tokens, dtype=torch.bool)
            tm = torch.zeros(num_tokens, dtype=torch.bool)
            cm[context_idx] = True
            tm[target_idx] = True
            context_masks.append(cm.view(self.grid_size, self.grid_size))
            target_masks.append(tm.view(self.grid_size, self.grid_size))
        return torch.stack(context_masks), torch.stack(target_masks)

    def forward(self, x):
        B = x.size(0)
        context_mask, target_mask = self.create_feature_masks(B)
        context_mask = context_mask.to(x.device)
        target_mask = target_mask.to(x.device)

        # Target features (stop gradient)
        with torch.no_grad():
            target_feat = self.target_encoder(x)  # [B,C,7,7]
            
        # Context features
        context_feat = self.context_encoder(x)  # [B,C,7,7]
        
        # Mask out target regions
        cm = context_mask.unsqueeze(1).float()
        context_masked = context_feat * cm
        
        # Predictor
        pred_feat = self.predictor(context_masked)
        
        # Loss only on target regions
        tm = target_mask.unsqueeze(1).float()
        loss = F.mse_loss(pred_feat * tm, target_feat * tm)
        return loss, pred_feat, target_feat, context_mask, target_mask

    @torch.no_grad()
    def update_target_encoder(self, momentum=0.99):
        # EMA update of target encoder from context encoder
        for tp, cp in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            tp.data = tp.data * momentum + cp.data * (1.0 - momentum)


class JEPAEncoder(nn.Module):
    def __init__(self, model_arch='resnet18', imgformat='rgb', proj_dim=256, weights=None):
        super().__init__()

        # Always start from vanilla ResNet; weights will be overwritten by JEPA weights
        base_model = getattr(models, model_arch)(weights=None)
        
        # # Validate input
        # if imgformat not in ["gray", "rgb"]:
        #     raise ValueError("imgformat must be 'gray' or 'rgb'")

        # # Load backbone
        # if weights:
        #     base_model = getattr(models, model_arch)(weights=weights)
        # else:
        #     base_model = getattr(models, model_arch)(weights=None)

        # Modify first conv if grayscale
        if imgformat == 'gray':
            original_conv = base_model.conv1
            base_model.conv1 = nn.Conv2d(
                in_channels=1,
                out_channels=original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )
            with torch.no_grad():
                base_model.conv1.weight[:,0] = original_conv.weight[:,0]

        # Keep encoder layers
        self.stem = nn.Sequential(base_model.conv1, base_model.bn1, base_model.relu)
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        # Projection head as in JEPA
        self.proj_head = nn.Sequential(
            nn.Conv2d(512, proj_dim, kernel_size=1),
            nn.BatchNorm2d(proj_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Collect intermediate outputs
        x0 = self.stem(x)          # [B,64,H/2,W/2]
        x1 = self.maxpool(x0)
        x1 = self.layer1(x1)       # [B,64,H/4,W/4]
        x2 = self.layer2(x1)       # [B,128,H/8,W/8]
        x3 = self.layer3(x2)       # [B,256,H/16,W/16]
        x4 = self.layer4(x3)       # [B,512,H/32,W/32]
        x4_proj = self.proj_head(x4)  # [B,proj_dim,H/32,W/32]
        return x4_proj, [x3, x2, x1, x0]  # skip features list

class JEPAUnetDecoder(nn.Module):
    def __init__(self, proj_dim=256, encoder_channels=[256,128,64,64], out_classes=1):
        super().__init__()

        # up from proj_dim (256) + skip 256
        self.upconv4 = nn.ConvTranspose2d(proj_dim, 256, kernel_size=2, stride=2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(proj_dim + encoder_channels[0], 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Conv2d(64, out_classes, kernel_size=1)

    def forward(self, x, skips, original_size):
        # x: x4_proj
        # skips: [x3, x2, x1, x0]
        # First block
        x = self.upconv4(x)
        x = torch.cat([x, skips[0]], dim=1)
        x = self.conv4(x)

        # Second block
        x = self.upconv3(x)
        x = torch.cat([x, skips[1]], dim=1)
        x = self.conv3(x)

        # Third block
        x = self.upconv2(x)
        x = torch.cat([x, skips[2]], dim=1)
        x = self.conv2(x)

        # Fourth block
        x = self.upconv1(x)
        x = torch.cat([x, skips[3]], dim=1)
        x = self.conv1(x)

        x = self.final_conv(x)
        x = nn.functional.interpolate(x, size=original_size, mode='bilinear', align_corners=False)
        return x

class JEPAUnet(nn.Module):
    def __init__(self, jepa_weights_path, proj_dim=256, out_classes=1, imgformat='rgb'):
        super().__init__()
        # Build encoder and load trained weights
        self.encoder = JEPAEncoder(proj_dim=proj_dim, imgformat=imgformat)
        state = torch.load(jepa_weights_path, map_location='cpu')
        self.encoder.load_state_dict(state, strict=False)
        print("JEPA encoder weights loaded .")

        # Freeze encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        print("JEPA encoder frozen.")

        # Build decoder
        self.decoder = JEPAUnetDecoder(proj_dim=proj_dim, out_classes=out_classes)

    def forward(self, x):
        original_size = (x.shape[2], x.shape[3])
        x4_proj, skips = self.encoder(x)   # get deepest + skip features
        seg = self.decoder(x4_proj, skips, original_size)
        return seg