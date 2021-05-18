import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .stylegan2_module import Style2ResidualBlock, Style2ResidualBlock1D, Style2ResidualBlock1DSrc, Style2ResidualBlock1DBeta


class GLU(nn.Module):
    ''' GLU block, do not split channels dimension'''

    def __init__(self,):
        super().__init__()

    def forward(self, x):

        return x * torch.sigmoid(x)


class AdaptiveInstanceNormalisation2D(nn.Module):

    def __init__(self, dim_in, dim_c):

        super().__init__()

        self.dim_in = dim_in

        self.norm = nn.InstanceNorm2d(dim_in, affine=False)

        self.fc_g = nn.Linear(dim_c, dim_in)
        self.fc_b = nn.Linear(dim_c, dim_in)

        #self.fc_g = nn.Linear(dim_c, dim_in)
        #self.fc_b = nn.Linear(dim_c, dim_in)

        #self.lat_linear = nn.Linear(2*dim_in, dim_c)

    def forward(self, x, c_src, c_trg):

        #x_flat = x.view(x.size(0), x.size(1), -1)
        #h = torch.cat([torch.mean(x_flat, 2), torch.std(x_flat, 2)], 1)
        #h = self.lat_linear(h)
        #h = F.relu(h)

        #dis_src_h = torch.sum(c_src * h, dim = -1, keepdim = True)
        #dis_trg_h = torch.sum(c_trg * h, dim = -1, keepdim = True)
        #src_att_weight = dis_src_h / (dis_src_h + dis_trg_h + 1e-10)
        #trg_att_weight = dis_trg_h / (dis_trg_h + dis_trg_h + 1e-10)

        #c =  src_att_weight * c_src + trg_att_weight * c_trg

        #c = torch.cat([h, c_trg - h], dim = 1)

        #c = torch.cat([c_src, c_trg - c_src], dim = -1)
        #c = torch.cat([c_src, c_trg], dim = -1)
        c = c_trg
        gamma = self.fc_g(c)  # * torch.sigmoid(self.gate_g(c))
        beta = self.fc_b(c)  # * torch.sigmoid(self.gate_b(c))

        gamma = gamma.view(gamma.size(0), gamma.size(1), 1, 1)
        beta = beta.view(beta.size(0), beta.size(1), 1, 1)

        # return (1 + gamma) * self.norm(x) + beta
        return gamma * self.norm(x) + beta


class AdaptiveInstanceNormalisation(nn.Module):
    """AdaIN Block."""

    def __init__(self, dim_in, dim_c):
        super(AdaptiveInstanceNormalisation, self).__init__()

        #self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.dim_in = dim_in
        #self.style_num = style_num

        #self.gamma_t = nn.Linear(dim_c, dim_in)
        #self.beta_t = nn.Linear(dim_c, dim_in)
        self.gamma_t = nn.Linear(2*dim_c, dim_in)
        self.beta_t = nn.Linear(2*dim_c, dim_in)

        #self.lat_linear = nn.Linear(2*dim_in, dim_c)

    def forward(self, x, c_src, c_trg):
        u = torch.mean(x, dim=2, keepdim=True)
        var = torch.mean((x - u) * (x - u), dim=2, keepdim=True)
        std = torch.sqrt(var + 1e-8)

        # width = x.shape[2]
        #x_flat = x.view(x.size(0), x.size(1), -1)
        #src_stats = torch.cat([var.squeeze(2), std.squeeze(2)], 1)
        #dyn_src_cond = self.lat_linear(src_stats)
        #dyn_c_src = F.relu(dyn_src_cond)

        #c = torch.cat([c_trg, c_trg - dyn_c_src], dim = -1)
        c = torch.cat([c_src, c_trg], dim=-1)
        #c = c_trg
        #gamma = self.gamma_t(c_trg) + torch.sigmoid(self.gam_gate_s(c)) * self.gamma_s(c_src)
        gamma = self.gamma_t(c)  # * torch.sigmoid(self.gate_gamma_t(c))
        gamma = gamma.view(-1, self.dim_in, 1)
        #beta = self.beta_t(c_trg) + torch.sigmoid(self.bet_gate_s(c)) * self.beta_s(c_src)
        beta = self.beta_t(c)  # * torch.sigmoid(self.gate_beta_t(c))
        beta = beta.view(-1, self.dim_in, 1)

        h = (x - u) / std
        h = h * gamma + beta
        #h = h * gamma + u
        return h


class ConditionalInstanceNormalisation(nn.Module):
    """CIN Block."""

    def __init__(self, dim_in, style_num):
        super(ConditionalInstanceNormalisation, self).__init__()

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        self.dim_in = dim_in
        self.style_num = style_num
        self.gamma = nn.Linear(style_num, dim_in)
        self.beta = nn.Linear(style_num, dim_in)

    def forward(self, x, c):
        u = torch.mean(x, dim=2, keepdim=True)
        var = torch.mean((x - u) * (x - u), dim=2, keepdim=True)
        std = torch.sqrt(var + 1e-8)

        # width = x.shape[2]

        gamma = self.gamma(c.to(self.device))
        gamma = gamma.view(-1, self.dim_in, 1)
        beta = self.beta(c.to(self.device))
        beta = beta.view(-1, self.dim_in, 1)

        h = (x - u) / std
        h = h * gamma + beta

        return h


class ResidualBlock2D(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out):
        super(ResidualBlock2D, self).__init__()
        self.conv_1 = nn.Conv2d(
            dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.adain_1 = AdaptiveInstanceNormalisation2D(dim_out, 128)
        self.glu_1 = nn.GLU(dim=1)

    def forward(self, x, c_src, c_trg):
        x_ = self.conv_1(x)
        x_ = self.adain_1(x_, c_src, c_trg)
        #x_ = torch.sigmoid(x_) * x_
        x_ = self.glu_1(x_)

        return x_


class ResidualBlockSplit(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out):
        super(ResidualBlockSplit, self).__init__()
        self.conv_1 = nn.Conv1d(
            dim_in, 2*dim_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.cin_1 = AdaptiveInstanceNormalisation(2 * dim_out, 128)
        self.glu_1 = nn.GLU(dim=1)

    def forward(self, x, c_src, c_trg):
        x_ = self.conv_1(x)
        x_ = self.cin_1(x_, c_src, c_trg)
        x_ = self.glu_1(x_)
        return x_


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""

    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.conv_1 = nn.Conv1d(
            dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.cin_1 = AdaptiveInstanceNormalisation(dim_out, 128)
        self.glu_1 = GLU()
        #self.relu = nn.LeakyReLU(0.2)

    def forward(self, x, c_src, c_trg):
        x_ = self.conv_1(x)
        x_ = self.cin_1(x_, c_src, c_trg)
        #x_ = torch.sigmoid(x_) * x_
        x_ = self.glu_1(x_)
        #x_ = self.relu(x_)
        return x_


class SEBlock(nn.Module):
    '''Squeeze and Excitation Block'''

    def __init__(self, in_dim, hid_dim):

        super().__init__()

        self.conv = nn.Conv1d(in_dim, in_dim, kernel_size=5,
                              stride=1, padding=2, bias=False)

        self.linear1 = nn.Linear(in_dim, hid_dim)
        self.relu1 = nn.ReLU()

        self.linear2 = nn.Linear(hid_dim, in_dim)

    def forward(self, x):
        '''
            x: input, shape: [b,c,t]
        '''
        conv_out = self.conv(x)

        mean = torch.mean(conv_out, dim=2)

        z = self.linear1(mean)
        z = self.relu1(z)

        z = self.linear2(z)
        z = torch.sigmoid(z)

        # residual
        out = x + conv_out * z.unsqueeze(2)

        return out


class SPEncoderPool1D(nn.Module):
    '''speaker encoder for adaptive instance normalization, add statistic pooling layer'''

    def __init__(self, num_speakers=4, spk_cls=False):

        super().__init__()
        self.cls = spk_cls

        '''
        self.down_sample_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=5, stride = 1, padding=2, bias=False),
            nn.LeakyReLU(0.02),
        )
        self.down_sample_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.02),
        )
        self.down_sample_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.02),
        )
        self.down_sample_4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.02),
        )
        self.down_sample_5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.02),
        )
        '''

        self.down_sample_1 = nn.Sequential(
            nn.Conv1d(in_channels=36, out_channels=256,
                      kernel_size=5, stride=1, padding=2, bias=False),
            nn.LeakyReLU(0.02),
        )
        self.down_sample_2 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256,
                      kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.02),
        )
        self.down_sample_3 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256,
                      kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.02),
        )
        self.down_sample_4 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.02),
        )
        self.down_sample_5 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.02),
        )
        #self.linear1 = nn.Linear(256, 128)

        self.unshared = nn.ModuleList()

        for _ in range(num_speakers):
            self.unshared += [nn.Linear(512, 128)]

        if self.cls:
            self.cls_layer = nn.Linear(128, num_speakers)

    def forward(self, x, trg_c, cls_out=False):

        x = x.squeeze(1)

        out = self.down_sample_1(x)

        out = self.down_sample_2(out)

        out = self.down_sample_3(out)

        out = self.down_sample_4(out)

        out = self.down_sample_5(out)

        #b,c,h,w = out.size()
        #out = out.view(b,c,h*w)
        out_mean = torch.mean(out, dim=2)
        out_std = torch.std(out, dim=2)

        out = torch.cat([out_mean, out_std], dim=1)

        #out = self.linear1(out)
        res = []
        for layer in self.unshared:

            res += [layer(out)]

        res = torch.stack(res, dim=1)

        idx = torch.LongTensor(range(x.size(0))).to(x.device)
        s = res[idx, trg_c.long()]

        if self.cls and cls_out:
            cls_out = self.cls_layer(s)
            return s, cls_out
        else:
            return s


class SPEncoderTDNNPool(nn.Module):
    '''speaker encoder for adaptive instance normalization, add tdnn + statistic pooling layer'''

    def __init__(self, num_speakers=4, dim_in=36, p_dropout=0.1, spk_cls=False):

        super().__init__()
        self.cls = spk_cls

        self.tdnn1 = nn.Conv1d(36, 512, kernel_size=5, dilation=1)
        self.bn_tdnn1 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn1 = nn.Dropout(p=p_dropout)

        self.tdnn2 = nn.Conv1d(512, 512, kernel_size=5, dilation=2)
        self.bn_tdnn2 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn2 = nn.Dropout(p=p_dropout)

        self.tdnn3 = nn.Conv1d(512, 512, kernel_size=7, dilation=3)
        self.bn_tdnn3 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn3 = nn.Dropout(p=p_dropout)

        self.tdnn4 = nn.Conv1d(512, 512, kernel_size=1, dilation=1)
        self.bn_tdnn4 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn4 = nn.Dropout(p=p_dropout)

        self.tdnn5 = nn.Conv1d(512, 1500, kernel_size=1, dilation=1)
        self.bn_tdnn5 = nn.BatchNorm1d(1500, momentum=0.1, affine=False)
        self.dropout_tdnn5 = nn.Dropout(p=p_dropout)

        self.fc1 = nn.Linear(3000, 512)
        self.bn_fc1 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_fc1 = nn.Dropout(p=p_dropout)

        self.fc2 = nn.Linear(512, 512)
        self.bn_fc2 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_fc2 = nn.Dropout(p=p_dropout)

        self.embedding_layer = nn.Linear(512, 128)

        if self.cls:
            self.cls_layer = nn.Linear(512, num_speakers)

    def forward(self, x, trg_c=None, cls_out=False):

        x = x.squeeze(1)

        x = self.dropout_tdnn1(self.bn_tdnn1(F.relu(self.tdnn1(x))))
        x = self.dropout_tdnn2(self.bn_tdnn2(F.relu(self.tdnn2(x))))
        x = self.dropout_tdnn3(self.bn_tdnn3(F.relu(self.tdnn3(x))))
        x = self.dropout_tdnn4(self.bn_tdnn4(F.relu(self.tdnn4(x))))
        x = self.dropout_tdnn5(self.bn_tdnn5(F.relu(self.tdnn5(x))))
        '''
        if self.training:
            shape = x.size()
            noise = torch.randn(shape).to(x.device)
            x += noise * 1e-5
        '''
        stats = torch.cat([x.mean(dim=2), x.std(dim=2)], dim=1)
        x = self.dropout_fc1(self.bn_fc1(F.relu(self.fc1(stats))))
        x = self.dropout_fc2(self.bn_fc2(F.relu(self.fc2(x))))

        s = self.embedding_layer(x)

        if self.cls and cls_out:
            cls_out = self.cls_layer(x)
            return s, cls_out
        else:
            return s


class SPEncoderPool(nn.Module):
    '''speaker encoder for adaptive instance normalization, add statistic pooling layer'''

    def __init__(self, num_speakers=4, spk_cls=False):

        super().__init__()
        self.cls = spk_cls

        self.down_sample_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=5,
                      stride=1, padding=2, bias=False),
            nn.LeakyReLU(0.02),
        )
        self.down_sample_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.02),
        )
        self.down_sample_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.02),
        )
        self.down_sample_4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.02),
        )
        self.down_sample_5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.02),
        )
        '''
        
        
        self.down_sample_1 = nn.Sequential(
            nn.Conv1d(in_channels=36, out_channels=256, kernel_size=5, stride = 1, padding=2, bias=False),
            nn.LeakyReLU(0.02),
        )
        self.down_sample_2 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.02),
        )
        self.down_sample_3 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.02),
        )
        self.down_sample_4 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.02),
        )
        self.down_sample_5 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.02),
        )
        '''
        self.linear1 = nn.Linear(512, 128)

        #self.unshared = nn.ModuleList()

        # for _ in range(num_speakers):
        #    self.unshared += [nn.Linear(512, 128)]

        if self.cls:
            self.cls_layer = nn.Linear(128, num_speakers)

    def forward(self, x, trg_c, cls_out=False):

        #x = x.squeeze(1)

        out = self.down_sample_1(x)

        out = self.down_sample_2(out)

        out = self.down_sample_3(out)

        out = self.down_sample_4(out)

        out = self.down_sample_5(out)

        b, c, h, w = out.size()
        out = out.view(b, c, h*w)
        out_mean = torch.mean(out, dim=2)
        out_std = torch.std(out, dim=2)

        out = torch.cat([out_mean, out_std], dim=1)

        s = self.linear1(out)
        '''
        res = []
        for layer in self.unshared:
            
            res += [layer(out)]

        res = torch.stack(res, dim = 1)
        
        idx = torch.LongTensor(range(x.size(0))).to(x.device)
        s = res[idx, trg_c.long()]
        '''
        if self.cls and cls_out:
            cls_out = self.cls_layer(s)
            return s, cls_out
        else:
            return s


class SPEncoder(nn.Module):
    '''speaker encoder for adaptive instance normalization'''

    def __init__(self, num_speakers=4, spk_cls=False):

        super().__init__()
        self.spk_cls = spk_cls
        self.down_sample_1 = nn.Sequential(
            nn.Conv1d(in_channels=36, out_channels=256,
                      kernel_size=5, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.02),
        )
        self.down_sample_2 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256,
                      kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.02),
        )
        self.down_sample_3 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256,
                      kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.02),
        )
        self.down_sample_4 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256,
                      kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.02),
        )
        self.down_sample_5 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.02),
        )
        '''
        
        
        self.down_sample_1 = nn.Sequential(
            nn.Conv1d(in_channels=36, out_channels=256, kernel_size=5, stride = 1, padding=2, bias=False),
            nn.LeakyReLU(0.02),
        )
        self.down_sample_2 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.02),
        )
        self.down_sample_3 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.02),
        )
        self.down_sample_4 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.02),
        )
        self.down_sample_5 = nn.Sequential(
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.02),
        )
        '''
        #self.linear1 = nn.Linear(512, 128)
        # if spk_cls:
        #    self.cls_layer = nn.Linear(128, num_speakers)
        self.unshared = nn.ModuleList()

        for _ in range(num_speakers):
            #self.unshared += [nn.Linear(256, 128)]
            self.unshared += [nn.Linear(512, 128)]

    def forward(self, x, trg_c, cls_out=False):

        x = x.squeeze(1)

        out = self.down_sample_1(x)

        out = self.down_sample_2(out)

        out = self.down_sample_3(out)

        out = self.down_sample_4(out)

        out = self.down_sample_5(out)

        #b,c,h,w = out.size()
        #out = out.view(b,c,h*w)
        #out = torch.mean(out, dim = 2)
        out_mean = torch.mean(out, dim=2)
        out_std = torch.std(out, dim=2)

        out = torch.cat([out_mean, out_std], dim=1)

        #out = self.linear1(out)
        res = []
        for layer in self.unshared:

            res += [layer(out)]

        res = torch.stack(res, dim=1)

        idx = torch.LongTensor(range(x.size(0))).to(x.device)
        s = res[idx, trg_c.long()]

        return s
        # if self.spk_cls and cls_out:
        #    cls_out = self.cls_layer(out)
        #    return out, cls_out
        # else:
        #    return out


class Generator2D(nn.Module):
    """Generator network."""

    def __init__(self, num_speakers=4, aff=True, res_block_name=''):
        super(Generator2D, self).__init__()
        # Down-sampling layers
        '''
        self.down_sample_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(3, 9), padding=(1, 4), bias=False),
            nn.GLU(dim=1)
        )
        self.down_sample_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(4, 8), stride=(2, 2), padding=(1, 3), bias=False),
            nn.InstanceNorm2d(num_features=256, affine=True, track_running_stats=True),
            nn.GLU(dim=1)
        )
        self.down_sample_3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(4, 8), stride=(2, 2), padding=(1, 3), bias=False),
            nn.InstanceNorm2d(num_features=512, affine=True, track_running_stats=True),
            nn.GLU(dim=1)
        )
        '''
        self.down_sample_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=256,
                      kernel_size=(3, 9), padding=(1, 4), bias=False),
            nn.GLU(dim=1)
        )
        self.down_sample_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(
                4, 8), stride=(2, 2), padding=(1, 3), bias=False),
            nn.InstanceNorm2d(num_features=512, affine=aff),
            nn.GLU(dim=1)
        )
        self.down_sample_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(
                4, 8), stride=(2, 2), padding=(1, 3), bias=False),
            nn.InstanceNorm2d(num_features=512, affine=aff),
            nn.GLU(dim=1)
        )
        # b 256 9 32
        # Down-conversion layers.
        '''
        self.down_conversion = nn.Sequential(
            nn.Conv1d(in_channels=2304,
                      out_channels=256,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.InstanceNorm1d(num_features=256, affine=aff)
        )
        '''
        # Bottleneck layers.
        #self.residual_1 = ResidualBlock2D(dim_in=256, dim_out=512)
        #self.residual_2 = ResidualBlock2D(dim_in=256, dim_out=512)
        #self.residual_3 = ResidualBlock2D(dim_in=256, dim_out=512)
        #self.residual_4 = ResidualBlock2D(dim_in=256, dim_out=512)
        #self.residual_5 = ResidualBlock2D(dim_in=256, dim_out=512)
        #self.residual_6 = ResidualBlock2D(dim_in=256, dim_out=512)
        #self.residual_7 = ResidualBlock2D(dim_in=256, dim_out=512)
        #self.residual_8 = ResidualBlock2D(dim_in=256, dim_out=512)
        #self.residual_9 = ResidualBlock2D(dim_in=256, dim_out=512)

        self.residual_1 = Style2ResidualBlock(dim_in=256, dim_out=256)
        self.residual_2 = Style2ResidualBlock(dim_in=256, dim_out=256)
        self.residual_3 = Style2ResidualBlock(dim_in=256, dim_out=256)
        self.residual_4 = Style2ResidualBlock(dim_in=256, dim_out=256)
        self.residual_5 = Style2ResidualBlock(dim_in=256, dim_out=256)
        self.residual_6 = Style2ResidualBlock(dim_in=256, dim_out=256)
        self.residual_7 = Style2ResidualBlock(dim_in=256, dim_out=256)
        self.residual_8 = Style2ResidualBlock(dim_in=256, dim_out=256)
        self.residual_9 = Style2ResidualBlock(dim_in=256, dim_out=256)
        '''
        # Up-conversion layers.
        self.up_conversion = nn.Conv1d(in_channels=256,
                                       out_channels=2304,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       bias=False)
        '''
        # Up-sampling layers.
        '''
        self.up_sample_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=256, affine=True, track_running_stats=True),
            nn.GLU(dim=1)
        )
        self.up_sample_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=128, affine=True, track_running_stats=True),
            nn.GLU(dim=1)
        )
        '''
        self.up_sample_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=512,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=512, affine=True),
            nn.GLU(dim=1)
        )
        self.up_sample_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=256,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(num_features=256, affine=True),
            nn.GLU(dim=1)
        )
        # Out.
        self.out = nn.Conv2d(in_channels=128, out_channels=1,
                             kernel_size=7, stride=1, padding=3, bias=False)

    def forward(self, x, c_src, c_trg):
        width_size = x.size(3)

        x = self.down_sample_1(x)
        x = self.down_sample_2(x)
        x = self.down_sample_3(x)

        #x = x.contiguous().view(-1, 2304, width_size // 4)
        #x = self.down_conversion(x)

        x = self.residual_1(x, c_src, c_trg)
        x = self.residual_2(x, c_src, c_trg)
        x = self.residual_3(x, c_src, c_trg)
        x = self.residual_4(x, c_src, c_trg)
        x = self.residual_5(x, c_src, c_trg)
        x = self.residual_6(x, c_src, c_trg)
        x = self.residual_7(x, c_src, c_trg)
        x = self.residual_8(x, c_src, c_trg)
        x = self.residual_9(x, c_src, c_trg)

        #x = self.up_conversion(x)
        #x = x.view(-1, 256, 9, width_size // 4)

        x = self.up_sample_1(x)
        x = self.up_sample_2(x)
        x = self.out(x)

        return x


class GeneratorSplit(nn.Module):
    """Generator network."""

    def __init__(self, num_speakers=4, aff=True, res_block_name='ResidualBlockSplit'):
        super(GeneratorSplit, self).__init__()
        # Down-sampling layers
        self.res_block_name = res_block_name
        self.down_sample_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=256,
                      kernel_size=(3, 9), padding=(1, 4), bias=False),
            nn.GLU(dim=1),
        )
        self.down_sample_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(
                4, 8), stride=(2, 2), padding=(1, 3), bias=False),
            nn.InstanceNorm2d(num_features=512, affine=aff),
            nn.GLU(dim=1)
        )
        self.down_sample_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(
                4, 8), stride=(2, 2), padding=(1, 3), bias=False),
            nn.InstanceNorm2d(num_features=512, affine=aff),
            nn.GLU(dim=1)
        )
        # Down-conversion layers.
        self.down_conversion = nn.Sequential(
            nn.Conv1d(in_channels=2304,
                      out_channels=256,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.InstanceNorm1d(num_features=256, affine=aff)
        )

        # Bottleneck layers.

        '''
        self.residual_1 = ResidualBlockSplit(dim_in=256, dim_out=256)
        self.residual_2 = ResidualBlockSplit(dim_in=256, dim_out=256)
        self.residual_3 = ResidualBlockSplit(dim_in=256, dim_out=256)
        self.residual_4 = ResidualBlockSplit(dim_in=256, dim_out=256)
        self.residual_5 = ResidualBlockSplit(dim_in=256, dim_out=256)
        self.residual_6 = ResidualBlockSplit(dim_in=256, dim_out=256)
        self.residual_7 = ResidualBlockSplit(dim_in=256, dim_out=256)
        self.residual_8 = ResidualBlockSplit(dim_in=256, dim_out=256)
        self.residual_9 = ResidualBlockSplit(dim_in=256, dim_out=256)
        '''

        '''
        self.residual_1 = Style2ResidualBlock1D(dim_in=256, dim_out=256)
        self.residual_2 = Style2ResidualBlock1D(dim_in=256, dim_out=256)
        self.residual_3 = Style2ResidualBlock1D(dim_in=256, dim_out=256)
        self.residual_4 = Style2ResidualBlock1D(dim_in=256, dim_out=256)
        self.residual_5 = Style2ResidualBlock1D(dim_in=256, dim_out=256)
        self.residual_6 = Style2ResidualBlock1D(dim_in=256, dim_out=256)
        self.residual_7 = Style2ResidualBlock1D(dim_in=256, dim_out=256)
        self.residual_8 = Style2ResidualBlock1D(dim_in=256, dim_out=256)
        self.residual_9 = Style2ResidualBlock1D(dim_in=256, dim_out=256)
        '''
        self.residual_1 = eval(self.res_block_name)(dim_in=256, dim_out=256)
        self.residual_2 = eval(self.res_block_name)(dim_in=256, dim_out=256)
        self.residual_3 = eval(self.res_block_name)(dim_in=256, dim_out=256)
        self.residual_4 = eval(self.res_block_name)(dim_in=256, dim_out=256)
        self.residual_5 = eval(self.res_block_name)(dim_in=256, dim_out=256)
        self.residual_6 = eval(self.res_block_name)(dim_in=256, dim_out=256)
        self.residual_7 = eval(self.res_block_name)(dim_in=256, dim_out=256)
        self.residual_8 = eval(self.res_block_name)(dim_in=256, dim_out=256)
        self.residual_9 = eval(self.res_block_name)(dim_in=256, dim_out=256)
        # Up-conversion layers.
        self.up_conversion = nn.Conv1d(in_channels=256,
                                       out_channels=2304,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       bias=False)

        # Up-sampling layers.

        self.up_sample_1 = nn.ConvTranspose2d(
            256, 2*256, kernel_size=4, stride=2, padding=1)
        self.up_in_1 = nn.InstanceNorm2d(2*256, affine=True)
        self.up_relu_1 = nn.GLU(dim=1)
        #self.up_relu_1 = nn.LeakyReLU(0.2)

        self.up_sample_2 = nn.ConvTranspose2d(
            256, 2 * 128, kernel_size=4, stride=2, padding=1)
        self.up_in_2 = nn.InstanceNorm2d(2 * 128, affine=True)
        self.up_relu_2 = nn.GLU(dim=1)
        #self.up_relu_2 = nn.LeakyReLU(0.2)

        # Out.
        self.out = nn.Conv2d(in_channels=128, out_channels=1,
                             kernel_size=7, stride=1, padding=3, bias=False)

    def forward(self, x, c_src, c_trg):
        width_size = x.size(3)

        x = self.down_sample_1(x)
        x = self.down_sample_2(x)
        x = self.down_sample_3(x)

        x = x.contiguous().view(-1, 2304, width_size // 4)
        x = self.down_conversion(x)

        x = self.residual_1(x, c_src, c_trg)
        x = self.residual_2(x, c_src, c_trg)
        x = self.residual_3(x, c_src, c_trg)
        x = self.residual_4(x, c_src, c_trg)
        x = self.residual_5(x, c_src, c_trg)
        x = self.residual_6(x, c_src, c_trg)
        x = self.residual_7(x, c_src, c_trg)
        x = self.residual_8(x, c_src, c_trg)
        x = self.residual_9(x, c_src, c_trg)

        x = self.up_conversion(x)
        x = x.view(-1, 256, 9, width_size // 4)

        x = self.up_sample_1(x)
        x = self.up_in_1(x)
        x = self.up_relu_1(x)

        x = self.up_sample_2(x)
        x = self.up_in_2(x)
        x = self.up_relu_2(x)

        x = self.out(x)

        return x


class Generator(nn.Module):
    """Generator network."""

    def __init__(self, num_speakers=4, aff=True, res_block_name=''):
        super(Generator, self).__init__()
        self.res_block_name = res_block_name
        # Down-sampling layers
        self.down_sample_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128,
                      kernel_size=(3, 9), padding=(1, 4), bias=False),
            nn.LeakyReLU(0.2),
            # GLU()
        )
        self.down_sample_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(
                4, 8), stride=(2, 2), padding=(1, 3), bias=False),
            nn.InstanceNorm2d(num_features=256, affine=aff),
            nn.LeakyReLU(0.2),
            # GLU()
        )
        self.down_sample_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(
                4, 8), stride=(2, 2), padding=(1, 3), bias=False),
            nn.InstanceNorm2d(num_features=256, affine=aff),
            nn.LeakyReLU(0.2),
            # GLU()
        )
        # Down-conversion layers.
        self.down_conversion = nn.Sequential(
            nn.Conv1d(in_channels=2304,
                      out_channels=256,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.InstanceNorm1d(num_features=256, affine=aff)
        )

        # Bottleneck layers.
        self.residual_1 = eval(self.res_block_name)(dim_in=256, dim_out=256)
        self.residual_2 = eval(self.res_block_name)(dim_in=256, dim_out=256)
        self.residual_3 = eval(self.res_block_name)(dim_in=256, dim_out=256)
        self.residual_4 = eval(self.res_block_name)(dim_in=256, dim_out=256)
        self.residual_5 = eval(self.res_block_name)(dim_in=256, dim_out=256)
        self.residual_6 = eval(self.res_block_name)(dim_in=256, dim_out=256)
        self.residual_7 = eval(self.res_block_name)(dim_in=256, dim_out=256)
        self.residual_8 = eval(self.res_block_name)(dim_in=256, dim_out=256)
        self.residual_9 = eval(self.res_block_name)(dim_in=256, dim_out=256)

        # Up-conversion layers.
        self.up_conversion = nn.Conv1d(in_channels=256,
                                       out_channels=2304,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       bias=False)

        # Up-sampling layers.

        self.up_sample_1 = nn.ConvTranspose2d(
            256, 256, kernel_size=4, stride=2, padding=1)
        self.up_in_1 = nn.InstanceNorm2d(256, affine=True)
        #self.up_relu_1 = GLU()
        self.up_relu_1 = nn.LeakyReLU(0.2)

        self.up_sample_2 = nn.ConvTranspose2d(
            256, 128, kernel_size=4, stride=2, padding=1)
        self.up_in_2 = nn.InstanceNorm2d(128, affine=True)
        #self.up_relu_2 = GLU()
        self.up_relu_2 = nn.LeakyReLU(0.2)

        # Out.
        self.out = nn.Conv2d(in_channels=128, out_channels=1,
                             kernel_size=7, stride=1, padding=3, bias=False)

    def forward(self, x, c_src, c_trg):
        width_size = x.size(3)

        x = self.down_sample_1(x)
        x = self.down_sample_2(x)
        x = self.down_sample_3(x)

        x = x.contiguous().view(-1, 2304, width_size // 4)
        x = self.down_conversion(x)

        x = self.residual_1(x, c_src, c_trg)
        x = self.residual_2(x, c_src, c_trg)
        x = self.residual_3(x, c_src, c_trg)
        x = self.residual_4(x, c_src, c_trg)
        x = self.residual_5(x, c_src, c_trg)
        x = self.residual_6(x, c_src, c_trg)
        x = self.residual_7(x, c_src, c_trg)
        x = self.residual_8(x, c_src, c_trg)
        x = self.residual_9(x, c_src, c_trg)

        x = self.up_conversion(x)
        x = x.view(-1, 256, 9, width_size // 4)

        x = self.up_sample_1(x)
        x = self.up_in_1(x)
        x = self.up_relu_1(x)

        x = self.up_sample_2(x)
        x = self.up_in_2(x)
        x = self.up_relu_2(x)

        x = self.out(x)

        return x


class Discriminator(nn.Module):
    """Discriminator network."""

    def __init__(self, num_speakers=10):
        super(Discriminator, self).__init__()

        self.num_speakers = num_speakers
        # Initial layers.
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=256,
                      kernel_size=(3, 3), stride=(1, 2), padding=1),
            nn.GLU(dim=1)
        )
        #self.conv1 = nn.Conv2d(1, 128, kernel_size= (3,3), stride = 1, padding= 1)
        #self.gate1 = nn.Conv2d(1, 128, kernel_size = 3, stride = 1, padding = 1)

        # Down-sampling layers.
        self.down_sample_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(
                3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.InstanceNorm2d(512, affine=True),
            nn.GLU(dim=1)
        )
        #self.down_sample_1 = DisDown(128, 256, kernel_size = 3, stride = 2, padding = 1)

        self.down_sample_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=(
                3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.InstanceNorm2d(1024, affine=True),
            nn.GLU(dim=1)
        )
        #self.down_sample_2 = DisDown(256, 512, kernel_size = 3, stride = 2, padding = 1)

        self.down_sample_3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=(
                3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.InstanceNorm2d(2048, affine=True),
            nn.GLU(dim=1)
        )
        #self.down_sample_3 = DisDown(512, 1024, kernel_size = 3, stride = 2, padding = 1)
        self.down_sample_4 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(
                1, 5), stride=(1, 2), padding=(0, 2), bias=False),
            nn.GLU(dim=1)
        )
        #self.down_sample_4 = DisDown(1024, 512, kernel_size = (1,5), stride = 1, padding = (0,2))
        # Fully connected layer.
        self.fully_connected = nn.Linear(
            in_features=512, out_features=num_speakers)
        #self.fully_connected = nn.Linear(in_features=512, out_features=512)

        # Projection.
        #self.projection = nn.Linear(self.num_speakers, 512)
        #self.projection_trg = nn.Linear(128, 512)
        #self.projection_src = nn.Linear(128, 512)
        #self.projection = nn.Linear(256, 512)

        #self.spk_emb = nn.Embedding(num_speakers, 128)

    def forward(self, x, c, c_):
        #c_onehot = torch.cat((c, c_), dim=1)
        #c_onehot = c_

        x = self.conv_layer_1(x)
        #spk_emb = self.spk_emb(c_.long())
        #print(f'x {x.size()} spk_emb {spk_emb.size()}')
        #x = x + spk_emb.unsqueeze(2).unsqueeze(3)
        #x_conv = self.conv1(x)
        #x_gate = self.gate1(x)
        #out = x_conv * torch.sigmoid(x_gate)

        x = self.down_sample_1(x)
        x = self.down_sample_2(x)
        x = self.down_sample_3(x)
        x = self.down_sample_4(x)

        b, c, h, w = x.size()
        x = x.view(b, c, h*w)
        x = torch.mean(x, dim=2)
        x = self.fully_connected(x)
        #x = torch.sigmoid(x)

        idx = torch.LongTensor(range(x.size(0))).to(x.device)

        x = x[idx, c_.long()]

        return x


'''
class PatchDiscriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, num_speakers=10):
        super(PatchDiscriminator, self).__init__()

        self.num_speakers = num_speakers
        # Initial layers.
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(0.2)
            #GLU()
        )
        #self.conv1 = nn.Conv2d(1, 128, kernel_size= (3,3), stride = 1, padding= 1)
        #self.gate1 = nn.Conv2d(1, 128, kernel_size = 3, stride = 1, padding = 1)

        # Down-sampling layers.
        self.down_sample_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.LeakyReLU(0.2)
            #nn.InstanceNorm2d(256, affine = True),
            #GLU()
        )
        #self.down_sample_1 = DisDown(128, 256, kernel_size = 3, stride = 2, padding = 1)

        self.down_sample_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.LeakyReLU(0.2)
            #nn.InstanceNorm2d(512, affine = True),
            #GLU()
        )
        #self.down_sample_2 = DisDown(256, 512, kernel_size = 3, stride = 2, padding = 1)
        
        self.down_sample_3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
            nn.LeakyReLU(0.2)
            #nn.InstanceNorm2d(1024, affine = True),
            #GLU()
        )
        #self.down_sample_3 = DisDown(512, 1024, kernel_size = 3, stride = 2, padding = 1)
        self.down_sample_4 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(1, 8), stride=(1, 1), padding=(0, 1, bias=False),
            nn.LeakyReLU(0.2)
            #GLU()
        )
        
        self.dis_conv = nn.Conv2d(512, num_speakers, kernel_size = 1, stride = 1, padding = 0, bias = False )

    def forward(self, x, c, c_):
        #c_onehot = torch.cat((c, c_), dim=1)
        #c_onehot = c_

        x = self.conv_layer_1(x)
        #x_conv = self.conv1(x)
        #x_gate = self.gate1(x)
        #out = x_conv * torch.sigmoid(x_gate)

        x = self.down_sample_1(x)
        x = self.down_sample_2(x)
        x = self.down_sample_3(x)
        x = self.down_sample_4(x)
        
        x = self.dis_conv(x)

        b, c, h, w = x.size()
        x = x.view(b,c, h*w)
        x = torch.mean(x, dim = 2)

        idx = torch.LongTensor(range(x.size(0))).to(x.device)

        x = x[idx, c_.long()]

        return x

'''


class PatchDiscriminator(nn.Module):
    """Discriminator network with PatchGAN."""

    def __init__(self, num_speakers=10):
        super(PatchDiscriminator, self).__init__()

        self.num_speakers = num_speakers
        # Initial layers.

        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128,
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )

        self.down_sample_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2)
        )

        self.down_sample_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2)
        )

        self.down_sample_3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2)
        )

        self.down_sample_4 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2)
        )

        self.dis_conv = nn.Conv2d(512, num_speakers, kernel_size=(
            1, 8), stride=1, padding=0, bias=False)

    def forward(self, x, c, c_, trg_cond=None):
        #c_onehot = torch.cat((c, c_), dim=1)
        #c_onehot = c_

        x = self.conv_layer_1(x)  # 128
        #x_conv = self.conv1(x)
        #x_gate = self.gate1(x)
        #out = x_conv * torch.sigmoid(x_gate)

        x = self.down_sample_1(x)

        x = self.down_sample_2(x)

        x = self.down_sample_3(x)

        x = self.down_sample_4(x)

        x = self.dis_conv(x)

        b, c, h, w = x.size()
        x = x.view(b, c)
        #x = x.view(b,c, h * w)
        #x = torch.mean(x, dim = 2)

        idx = torch.LongTensor(range(x.size(0))).to(x.device)

        x = x[idx, c_.long()]

        return x
