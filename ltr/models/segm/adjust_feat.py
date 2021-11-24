import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))

def conv_no_relu(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes))


class feat_adjust(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(feat_adjust, self).__init__()

        # search region nodes linear transformation
        self.segment0 = conv(in_channel, out_channel, kernel_size=1, padding=0)
        self.segment1 = conv_no_relu(out_channel, out_channel)
        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, xf, zf, train_masks):
        f_test = self.segment1(self.segment0(xf))

        f_train0 = self.segment1(self.segment0(zf))
        f_train = f_train0.view(*f_train0.shape[:2], -1)


        # reshape mask to the feature size
        mask_pos = F.interpolate(train_masks, size=(f_train0.shape[-2], f_train0.shape[-1]))
        mask_neg = 1 - mask_pos
        mask_pos = mask_pos.view(*mask_pos.shape[:2], -1) #B, C, H*W
        mask_neg = mask_neg.view(*mask_neg.shape[:2], -1)


        return f_test, f_train, mask_pos, mask_neg, f_train0
