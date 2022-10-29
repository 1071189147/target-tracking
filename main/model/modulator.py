import torch.nn as nn
import torchvision
import torch
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np
class DCA(nn.Module):

    def __init__(self, in_channels, reduction = 2, cda=False):
        super().__init__()

        self.cda = cda

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.weighting = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=in_channels // reduction,
                      kernel_size=1,
                      padding=0,
                      stride=1, ),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels // reduction,
                      out_channels=in_channels,
                      kernel_size=1,
                      padding=0,
                      stride=1, )
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w1 = self.avgpool(x)
        w2 = self.maxpool(x)

        w1 = self.weighting(w1)
        w2 = self.weighting(w2)

        w = self.sigmoid(w1 + w2)

        return x * w.expand_as(x)
def feature_visualization(x, n=256,name='h'):
    """
    x:              Features to be visualized
    module_type:    Module type
    stage:          Module stage within model
    n:              Maximum number of feature maps to plot
    save_dir:       Directory to save results
    """
    batch, channels, height, width = x.shape  # batch, channels, height, width
    if height > 1 and width > 1:
        f = '/data2/wcx/fpndual/{}.png'.format(name)  # filename

        blocks = torch.chunk(x[0].cpu(), channels, dim=0)  # select batch index 0, block by channels
        n = min(n, channels)  # number of plots
        fig, ax = plt.subplots(math.ceil(n / 8), 8, tight_layout=True)  # 8 rows x n/8 cols
        ax = ax.ravel()
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        for i in range(n):
            ax[i].imshow(blocks[i].squeeze())  # cmap='gray'
            ax[i].axis('off')
        plt.title('Features')
        plt.savefig(f, dpi=300, bbox_inches='tight')
        plt.close()
        np.save(str(f.with_suffix('.npy')), x[0].cpu().numpy())  # npy save
class Conv_Inter(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.module = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=1, padding=0, bias=True)
        )

    def forward(self, srcs):
        # srcs: List of Tensors
        return [self.module(x) for x in srcs]
class Modulator(nn.Module):
    def __init__(self, channels=256, roi_out_size=3, roi_sample_num = 2, strides=[4,8,16,32,64,128],featmap_num=6):
        super(Modulator, self).__init__()
        for i in range(3):
            beta = nn.Parameter(1.0 * torch.ones((256 , 1, 1)), requires_grad=True)
            setattr(self, "beta_%d" % i, beta)
        self.xiuzheng = Conv_Inter()
        # self.attention = nn.ModuleList([DCA(256),DCA(256),DCA(256)])

    def compute_corr_losses(self, pred_lbs, gt_lbs):
        eps = 1e-5
        x = pred_lbs.reshape(-1)
        target = gt_lbs.reshape(-1)
        intersection = (x * target).sum()
        union = (x ** 2.0).sum() + (target ** 2.0).sum() + eps
        loss = 1. - (2 * intersection / union)
        return loss

    def get_label_map(self, boxes, H, W, bs, device="cuda"):
        """boxes: (bs, 4)"""
        boxes_xyxy = torch.round(boxes).int()
        labels = torch.zeros((bs, 1, H, W), dtype=torch.float32, device=device)
        for b in range(bs):
            x1, y1, x2, y2 = boxes_xyxy[b][0]
            x1, y1 = max(0, x1), max(0, y1)
            try:
                labels[b, 0, y1:y2, x1:x2] = 1.0
            except:
                print("too small bounding box")
                pass
        return labels  # (bs, 1, H, W)
    def forward(self, feats_z, gt_bboxes_z,feats_x,box_x=None):
        if box_x != None:
            # feats_z[0], feats_z[0] =  self.xiuzheng([feats_z[0],feats_z[0]])#juanji
            feats_z[0], feats_x[0] = self.xiuzheng([feats_z[0], feats_x[0]])

            bs = feats_x[0].shape[0]
            simi_mat = torch.bmm(feats_z[0].flatten(-2).transpose(-1, -2), feats_x[0].flatten(-2))
            trans_mat_01 = torch.softmax(simi_mat, dim=1)
            gt_lbs_0 = F.interpolate(self.get_label_map(gt_bboxes_z, 384, 384, bs), scale_factor=1 / 4, mode="bilinear",align_corners=False).flatten(-2).cuda()
            pred_lbs1 = torch.bmm(gt_lbs_0, trans_mat_01).view(bs, 1,96, 96)
            pred_lbs1_ms = (pred_lbs1, F.interpolate(pred_lbs1, scale_factor=1 / 2, mode="bilinear", align_corners=False),F.interpolate(pred_lbs1, scale_factor=1 / 4, mode="bilinear", align_corners=False)) #4,8,16
            gt_lbs_1 = F.interpolate(self.get_label_map(box_x, 384, 384, bs), scale_factor=1 / 4, mode="bilinear",align_corners=False).flatten(-2).cuda()
            for i in range(3):
                beta = getattr(self, "beta_%d" % i)
                t = pred_lbs1_ms[i] * beta
                feats_x[i] = feats_x[i] + t
                # feats_x[i] = self.attention[i](feats_x[i])
            loss = self.compute_corr_losses(pred_lbs1, gt_lbs_1)

            return feats_x, loss
        else:
            tmp1, feats_x[0] = self.xiuzheng([feats_z[0], feats_x[0]])  # juanji
            # feats_z[0], feats_z[0] = self.xiuzheng([feats_z[0], feats_z[0]])#juanjiquanzhong
            bs = feats_x[0].shape[0]
            simi_mat = torch.bmm(tmp1.flatten(-2).transpose(-1, -2), feats_x[0].flatten(-2))
            trans_mat_01 = torch.softmax(simi_mat, dim=1)
            gt_lbs_0 = F.interpolate(self.get_label_map(gt_bboxes_z, 384, 384, bs), scale_factor=1 / 4, mode="bilinear",
                                     align_corners=False).flatten(-2).cuda()
            pred_lbs1 = torch.bmm(gt_lbs_0, trans_mat_01).view(bs, 1, 96, 96)
            pred_lbs1_ms = (
                pred_lbs1, F.interpolate(pred_lbs1, scale_factor=1 / 2, mode="bilinear", align_corners=False),
                F.interpolate(pred_lbs1, scale_factor=1 / 4, mode="bilinear", align_corners=False))  # 4,8,16
            for i in range(3):
                beta = getattr(self, "beta_%d" % i)
                t = pred_lbs1_ms[i] * beta
                feats_x[i] = feats_x[i] + t
                # feats_x[i] = getattr(self, "pam_%d" % i)(feats_x[i])
            return feats_x, 1
            #



