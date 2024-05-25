import torch
import torch.nn as nn
import torch.nn.functional as F
from ops.dcn.deform_conv import ModulatedDeformConv

# ==========
# Spatio-temporal deformable fusion module
# ==========

class STDF(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, base_ks=3, deform_ks=3):
        """
        Args:
            in_nc: num of input channels.输入通道数
            out_nc: num of output channels.输出通道数
            nf: num of channels (filters) of each conv layer.每个卷积层的通道/卷积核数量
            nb: num of conv layers.卷积层数量
            deform_ks: size of the deformable kernel.可变形卷积核的尺寸
        """
        super(STDF, self).__init__()

        # stdf:
        #     in_nc: 7  # 1 for Y
        #     out_nc: 64
        #     nf: 32  # num of feature maps
        #     nb: 3  # num of conv layers
        #     base_ks: 3
        #     deform_ks: 3  # size of the deformable kernel

        self.nb = nb
        self.in_nc = in_nc
        self.deform_ks = deform_ks
        self.size_dk = deform_ks ** 2

        # in_conv 输入的第一层特征提取模块，输入通道数7，输出通道数32
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_nc, nf, base_ks, padding=base_ks//2),
            nn.ReLU(inplace=True)
        )
        # nb = 3,构建2层的下采样模块dn_conv和2层的上采样模块up_conv
        for i in range(1, nb):
            setattr(
                self, 'dn_conv{}'.format(i), nn.Sequential(
                    nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks//2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(nf, nf, base_ks, padding=base_ks//2),
                    nn.ReLU(inplace=True)
                )
            )
            setattr(
                self, 'up_conv{}'.format(i), nn.Sequential(
                    nn.Conv2d(2*nf, nf, base_ks, padding=base_ks//2),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True)
                )
            )
        # 无关紧要的conv，用来作特征底层特征的concatenate
        self.tr_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, base_ks, padding=base_ks//2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, padding=base_ks//2),
            nn.ReLU(inplace=True)
        )

        # regression head
        # why in_nc*3*size_dk(3*3=9)?
        #   in_nc: each map use individual offset and mask 7
        #   2*size_dk: 2 coordinates for each point (每个点偏移的坐标(x,y))
        #   1*size_dk: 1 confidence (attention) score for each point (每个点的权重(注意力)分数)
        self.offset_mask = nn.Conv2d( # output channel :7*3*9 = 189
            nf, in_nc*3*self.size_dk, base_ks, padding=base_ks//2
        )

        # 深度可分离卷积实现
        # self.offset_mask = nn.Sequential(
        #     # deep wise conv
        #     nn.Conv2d(in_channels=nf, out_channels=nf,kernel_size=base_ks,stride=1,padding=base_ks // 2,groups=nf),
        #     # point wise conv
        #     nn.Conv2d(nf, in_nc * 3 * self.size_dk, 1, padding=0)
        # )

        # deformable conv
        # notice group=in_nc, i.e., each map use individual offset and mask
        self.deform_conv = ModulatedDeformConv(
            in_nc, out_nc, deform_ks, padding=deform_ks//2, deformable_groups=in_nc
        )

    def forward(self, inputs):
        nb = self.nb
        in_nc = self.in_nc
        n_off_msk = self.deform_ks * self.deform_ks

        # 改进点：改变输入inputs,用于计算offset_mask的inputs变为与目标帧的残差
        offsetmask_inputs = inputs - inputs[:,in_nc//2,:,:].unsqueeze(1)

        # feature extraction (with downsampling)
        out_lst = [self.in_conv(offsetmask_inputs)]  # record feature maps for skip connections
        for i in range(1, nb):
            dn_conv = getattr(self, 'dn_conv{}'.format(i))
            out_lst.append(dn_conv(out_lst[i - 1]))
        # trivial conv
        out = self.tr_conv(out_lst[-1]) # out_lst[-1] shape:[64,32,32,32]
        # feature reconstruction (with upsampling)
        for i in range(nb - 1, 0, -1):
            up_conv = getattr(self, 'up_conv{}'.format(i))
            out = up_conv(
                torch.cat([out, out_lst[i]], 1) # out shape :[64,32,128,128]
            )

        # compute offset and mask
        # offset: conv offset
        # mask: confidence

        # off_msk shape :[64,189,128,128]
        off_msk = self.offset_mask(self.out_conv(out))
        # off shape :[64,126,128,128]，前126个channel作为offset
        off = off_msk[:, :in_nc*2*n_off_msk, ...]
        # msk shape :[64,63,128,128],对off_msk后63个channel作Sigmoid计算
        msk = torch.sigmoid(
            off_msk[:, in_nc*2*n_off_msk:, ...]
        )

        # inputs shape :[64,7,128,128]
        # fused_feat shape :[64,64,128,128]

        # perform deformable convolutional fusion
        fused_feat = F.relu(
            self.deform_conv(inputs, off, msk), 
            inplace=True
        )

        return fused_feat


# ==========
# Quality enhancement module
# ==========

class PlainCNN(nn.Module):
    def __init__(self, in_nc=64, nf=48, nb=8, out_nc=3, base_ks=3):
        """
        Args:
            in_nc: num of input channels from STDF.
            nf: num of channels (filters) of each conv layer.
            nb: num of conv layers.
            out_nc: num of output channel. 3 for RGB, 1 for Y.
        """
        super(PlainCNN, self).__init__()

        # qenet:
        #     in_nc: 64  # = out_nc of stdf
        #     out_nc: 1  # 1 for Y
        #     nf: 48
        #     nb: 8
        #     base_ks: 3

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_nc, nf, base_ks, padding=1),
            nn.ReLU(inplace=True)
        )

        hid_conv_lst = []
        for _ in range(nb - 2):
            hid_conv_lst += [
                nn.Conv2d(nf, nf, base_ks, padding=1),
                nn.ReLU(inplace=True)
            ]
        self.hid_conv = nn.Sequential(*hid_conv_lst)

        self.out_conv = nn.Conv2d(nf, out_nc, base_ks, padding=1)

    def forward(self, inputs):
        out = self.in_conv(inputs)
        out = self.hid_conv(out)
        out = self.out_conv(out)
        return out


# ==========
# MFVQE network
# ==========

class MFVQE(nn.Module):
    """STDF -> QE -> residual.
    
    in: (B T C H W)
    out: (B C H W)
    """
    def __init__(self, opts_dict):
        """
        Arg:
            opts_dict: network parameters defined in YAML.
        """
        super(MFVQE, self).__init__()

        self.radius = opts_dict['radius']     # radius: 3
        self.input_len = 2 * self.radius + 1  # total num of input frame = 2 * radius + 1
        self.in_nc = opts_dict['stdf']['in_nc']

        # stdf:
        #     in_nc: 1*(2*3+1)  # 1 for Y
        #     out_nc: 64
        #     nf: 32  # num of feature maps
        #     nb: 3  # num of conv layers
        #     base_ks: 3
        #     deform_ks: 3  # size of the deformable kernel

        self.ffnet = STDF(
            in_nc= self.in_nc * self.input_len, 
            out_nc=opts_dict['stdf']['out_nc'], 
            nf=opts_dict['stdf']['nf'], 
            nb=opts_dict['stdf']['nb'], 
            deform_ks=opts_dict['stdf']['deform_ks']
        )

        # qenet:
        #     in_nc: 64  # = out_nc of stdf
        #     out_nc: 1  # 1 for Y ,将增强Y与原来的U、V拼接起来作为增强视频
        #     nf: 48
        #     nb: 8
        #     base_ks: 3

        self.qenet = PlainCNN(
            in_nc=opts_dict['qenet']['in_nc'],  
            nf=opts_dict['qenet']['nf'], 
            nb=opts_dict['qenet']['nb'], 
            out_nc=opts_dict['qenet']['out_nc']
        )

    def forward(self, x): # 输入x是7个视频帧序列cat在一起的
        out = self.ffnet(x) # 帧对齐的out shape :[64,64,128,128]
        out = self.qenet(out) # 质量增强的out shape :[64,1,128,128],这部分得到的是目标帧的残差res
        # e.g., B C=[B1 B2 B3 R1 R2 R3 G1 G2 G3] H W, B C=[Y1 Y2 Y3] H W or B C=[B1 ... B7 R1 ... R7 G1 ... G7] H W
        frm_lst = [self.radius + idx_c * self.input_len for idx_c in range(self.in_nc)] # 计算要增强的是7帧中的中间帧，即第3帧 0 1 2 [3] 4 5 6
        out += x[:, frm_lst, ...]  # res: add middle frame
        return out
