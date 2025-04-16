import torch
from einops import rearrange
from torch import nn,einsum
from torch.nn.init import trunc_normal_
import torch.nn.functional as F
LidarChannel = 2
HSIChannel = 64
EMBED_DIM =16
Lidar_EMBED_DIM = 16
#窗口尺寸
window_size1 = 5
window_size2 = 7
window_size3 = 9
BAND = 64
CLASS_NUM = 11
linear_num = 128
FM = 24

class Mlp(nn.Module):
    def __init__(self, in_features , hidden_feature = None ,out_feature = None , drop = 0.1):
        super().__init__()
        out_feature = out_feature or in_features
        hidden_feature = hidden_feature or in_features

        self.fc1 = nn.Linear(in_features , hidden_feature)
        self.fc2 = nn.Linear(hidden_feature,hidden_feature)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_feature, out_feature)
        self.drop1 = nn.Dropout(drop)
        self.drop2 = nn.Dropout(drop)
        self.drop4 = nn.Dropout(drop)

    def forward(self,x):
        x = self.fc1(x)
        x = self.act1(x)

        x = self.fc2(x)
        x = self.act2(x)

        x = self.fc4(x)
        x = self.drop4(x)
        return x

class TempConv2d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1, ):
        super(TempConv2d, self).__init__()
        self.depthwise = torch.nn.Conv2d(in_channels,
                                         in_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=in_channels)
        self.bn = torch.nn.BatchNorm2d(in_channels)
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x

class Spa(nn.Module):
    def __init__(self, dim, segment_dim=8 , tmp=7 , C=3 ,qkv_bias = False , proj_drop = 0.):
        super().__init__()
        self.segment_dim = segment_dim

        self.tmp = tmp
        dim2 = C * tmp
        self.mlp_h = nn.Linear(dim2, dim2, bias = qkv_bias)
        self.mlp_w = nn.Linear(dim2, dim2 , bias = qkv_bias)
        self.mlp_c = nn.Linear(dim, dim, bias = qkv_bias)

        #init  weight problem
        self.reweight = Mlp(dim , dim*2, dim * 3)

        self.proj = nn.Linear(dim , dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B,T,H,W,C = x.shape

        S = C // self.segment_dim
        tmp = self.tmp

        #h
        h = x.transpose(3,2).reshape(B ,T,H*W//tmp ,tmp, self.segment_dim,S).permute(0, 1, 2, 4, 3, 5).reshape(B , T ,H*W//tmp, self.segment_dim , tmp*S)
        h = self.mlp_h(h).reshape(B, T, H*W//tmp, self.segment_dim, tmp , S).permute(0, 1, 2, 4, 3, 5).reshape(B, T, H, W, C).transpose(3, 2)

        #w
        w = x.reshape(B ,T ,H*W //tmp, tmp , self.segment_dim, S).permute(0, 1, 2, 4, 3, 5).reshape(B, T, H*W//tmp, self.segment_dim , tmp*S)
        w = self.mlp_w(w).reshape(B, T, H*W//tmp, self.segment_dim, tmp , S).permute(0, 1, 2, 4, 3, 5).reshape(B, T, H, W, C)

        #c
        c = self.mlp_c(x)

        a = (h + w + c).permute(0,4, 1, 2, 3).flatten(2).mean(2)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim = 0).unsqueeze(2).unsqueeze(2).unsqueeze(2)

        x = h * a[0] + w * a[1] + c * a[2]

        x = self.proj(x)
        return x

class Spe(nn.Module):
    def __init__(self, dim , segment_dim, band, C, qkv_bias = False , proj_drop = 0.):
        super().__init__()

        self.segment_dim = segment_dim
        dim2 = band * C

        self.mlp_t = nn.Linear(dim2,dim2,bias = qkv_bias)

        self.proj = nn.Linear(dim,dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self,x):
        B, T, H ,W, C = x.shape
        #计算M
        S = C // self.segment_dim
        t = x.reshape(B, T, H, W, self.segment_dim, S).permute(0, 4, 2, 3, 1, 5).reshape(B, self.segment_dim, H, W, T * S)
        t = self.mlp_t(t).reshape(B, self.segment_dim, H, W, T, S).permute(0, 4, 2, 3, 1, 5).reshape(B, T, H, W, C)

        x = t

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Attention(nn.Module):
    def __init__(self,dim,band,segment_dim,dropout=0.1):
        super(Attention, self).__init__()
        self.to_q = TempConv2d(dim*band,dim*band*segment_dim)
        self.to_k = TempConv2d(dim*band,dim*band*segment_dim)
        self.to_v = TempConv2d(dim*band,dim*band*segment_dim)
        # self.norm = nn.LayerNorm(dim*band)
        self.heads = segment_dim
        self.norm = nn.LayerNorm(dim)
        self.dim = dim
        self.scale = dim ** -0.5
        self.feed = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim *  2, dim),
            nn.Dropout(dropout)
        )
    def forward(self,x):

        q = self.to_q(x)
        head = self.dim
        q = rearrange(q,'b (h w) i j  -> b h (i j) w', w=head)

        v = self.to_v(x)
        v = rearrange(v, 'b (h w) i j  -> b h (i j) w', w=head)

        k = self.to_k(x)
        k = rearrange(k, 'b (h w) i j  -> b h (i j) w', w=head)

        dots = einsum('b h i d  , b h j d  -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = self.norm(out)
        out = self.feed(out)


        return out

class LidarTransBlock(nn.Module):
    #dim:25 seg:5
    def __init__(self,dim,segment_dim,tmp,band,C,siz,mlp_ratio=4.,qkv_bias = False,dropout = 0.1):
        super().__init__()
        self.attention1 = Attention(dim,band,segment_dim,dropout=0.1)
        self.attention2 = Attention(dim,siz,segment_dim,dropout=0.1)
        self.attention3 = Attention(dim,siz,segment_dim,dropout=0.1)
        self.linear = nn.Linear(dim*segment_dim,dim*segment_dim)
        self.linears = nn.Linear(dim*segment_dim,dim)
        self.head = segment_dim

    def forward(self,x):
        #x.shape:[2860,2,5,5,25]
        b,band,h,w,seg = x.shape
        x1 = x
        x2 = x
        head = self.head
        x = rearrange(x,'b i h w j -> b (i j) h w')
        out = self.attention1(x)
        out = rearrange(out,'b (i k) (h w) j -> b i h w (j k)',j = seg,h=h, k= head)


        x1 = rearrange(x1,'b i h w j -> b (h j) i w')
        out2 = self.attention2(x1)
        out2 = rearrange(out2, 'b (h k) (i w) j -> b i h w (j k)',j =seg,w = w,k = head)


        x2 = rearrange(x2,'b i h w j -> b (w j) i h',j=seg)
        out3 = self.attention3(x2)
        out3 = rearrange(out3, 'b (w k) (i h) j -> b i h w (j k)',j = seg,h=h,k=head)

        out =  out + out2 + out3
        return out

class HSIBlock(nn.Module):
    def __init__(self, dim, segment_dim,tmp, band,C,mlp_ratio=2., qkv_bias=False,
                 drop_path=0., act_layer=nn.ReLU, norm_layer=nn.LayerNorm, skip_lam=1.0,sz = 0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.s_norm1 = norm_layer(dim)
        self.s_fc = Spe(dim, segment_dim,band,C,qkv_bias=qkv_bias)
        self.fc = Spa(dim, segment_dim=segment_dim, tmp=tmp, C=C,qkv_bias=qkv_bias)


        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,hidden_feature=mlp_hidden_dim ,act_layer=act_layer)
        self.skip_lam = skip_lam

    def forward(self, x):
        x = self.s_fc(self.s_norm1(x))
        x =  x + self.fc(self.norm1(x))
        x =  x + self.mlp(self.norm2(x))
        return x



class HsiBLock(nn.Module):
    def __init__(self,Patch,BAND,CLASSES_NUM, layers, embed_dims, segment_dim ,siz):
        super().__init__()

        tmp = Patch
        qkv_bias = True
        C = int(embed_dims / segment_dim)

        self.num_classes = CLASSES_NUM
        self.blocks1 = nn.ModuleList([])
        for i in range(layers):
            self.blocks1.append(
                HSIBlock(embed_dims,segment_dim,tmp = tmp,band = BAND,C=C,qkv_bias = qkv_bias, sz=siz)
            )

    def forward(self,x,h,w,hh,ww,windowsz):
        #边缘填充大小
        endx = h + (windowsz - h % windowsz) % windowsz
        endy = w + (windowsz - w % windowsz) % windowsz
        shffule_x = x.permute(0,1,3,2,4)
        for blk in self.blocks1:
            x = blk(x)
            shffule_x = blk(shffule_x)
        B, T, H, W, C = x.shape

        x = x.permute(0,2,3,1,4)
        x = rearrange(x,'(h w) b0 b1 t c -> (h b0) (w b1) t c',h = hh , w = ww, b0 = windowsz,b1 = windowsz,t = T,c = C)
        x1 = x.reshape(endx,endy, -1)
        shffule_x = shffule_x.permute(0,3,2,1,4)
        shffule_x = rearrange(shffule_x, '(h w) b0 b1 t c -> (h b0) (w b1) t c', h=hh, w=ww, b0=windowsz, b1=windowsz, t=T, c=C)

        shffule_x1 = shffule_x.reshape(endx, endy, -1)
        x1 = torch.cat([shffule_x1,x1],dim=-1)
        x1 = x1[:h,:w,:]
        return x1


class LidarBLock(nn.Module):
    def __init__(self, Patch, BAND, CLASSES_NUM, layers, embed_dims, segment_dim, siz):
        super().__init__()

        tmp = Patch
        qkv_bias = True
        C = int(embed_dims / segment_dim)

        self.num_classes = CLASSES_NUM
        # hsi block
        self.blocks1 = nn.ModuleList([])
        for i in range(layers):
            self.blocks1.append(
                LidarTransBlock(embed_dims, segment_dim, siz=siz, tmp=tmp, band=BAND, C=C, qkv_bias=qkv_bias),
            )

    def forward(self, x, h, w, hh,ww,windowsz):
        # 边缘填充大小
        endx = h + (windowsz - h % windowsz) % windowsz
        endy = w + (windowsz - w % windowsz) % windowsz
        shffule_x = x.permute(0,1,3,2,4)
        for blk in self.blocks1:
            x = blk(x)
            shffule_x = blk(shffule_x)
        B, T, H, W, C = x.shape

        x = x.permute(0,2,3,1,4)
        x = rearrange(x,'(h w) b0 b1 t c -> (h b0) (w b1) t c',h = hh , w = ww, b0 = windowsz,b1 = windowsz,t = T,c = C)
        x1 = x.reshape(endx,endy, -1)
        shffule_x = shffule_x.permute(0,3,2,1,4)
        shffule_x = rearrange(shffule_x, '(h w) b0 b1 t c -> (h b0) (w b1) t c', h=hh, w=ww, b0=windowsz, b1=windowsz, t=T, c=C)

        shffule_x1 = shffule_x.reshape(endx, endy, -1)
        x1 = torch.cat([shffule_x1,x1],dim=-1)
        x1 = x1[:h,:w,:]
        return x1

class PatchEmbed(nn.Module):
    def __init__(self, in_chans = 1, embed_dim = EMBED_DIM):
        super().__init__()
        self.proj1 = nn.Conv3d(in_chans, embed_dim // 2, (3, 3,7), (1, 1, 1), (1, 1, 3))

        self.norm1 = nn.BatchNorm3d(embed_dim // 2)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.act5 = nn.ReLU()
        self.proj2 = nn.Conv3d(embed_dim // 2, embed_dim//4, (3, 3, 7),(1, 1, 1), (1, 1, 3))
        self.norm2 = nn.BatchNorm3d(embed_dim // 4)

        self.proj5 = nn.Conv3d(embed_dim //4 , embed_dim //2 ,(3,3,7),(1,1,1),(1,1,3))
        self.proj6 = nn.Conv3d(embed_dim//2,embed_dim ,(3,3,7),(1,1,1),(1,1,3))

    def forward(self, x):
        x = self.proj1(x)
        x = self.norm1(x)
        x = self.act1(x)

        x = self.proj2(x)
        x = self.norm2(x)
        x = self.act2(x)

        x = self.proj5(x)
        x = self.norm1(x)
        x = self.act5(x)

        x = self.proj6(x)
        return x

class PPatchEmbed(nn.Module):
    def __init__(self, in_chans = 1, embed_dim =Lidar_EMBED_DIM):
        super().__init__()
        #nn.Conv3d(inp, oup, (3, kernel_size, kernel_size), (1, stride, stride), (1, padding, padding))
        self.proj1 = nn.Conv3d(in_chans, embed_dim // 2, (3, 3,1), (1, 1, 1), (1, 1, 0))

        #nn.Conv3d(inp, oup, (1, kernel_size, kernel_size), (1, stride, stride), (0, padding, padding))
        self.norm1 = nn.BatchNorm3d(embed_dim // 2)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.act3 = nn.ReLU()
        self.act4 = nn.ReLU()
        self.proj2 = nn.Conv3d(embed_dim // 2, embed_dim//4, (3, 3, 1),(1, 1, 1), (1, 1, 0))
        self.norm2 = nn.BatchNorm3d(embed_dim // 4)
        self.norm3 = nn.BatchNorm3d(embed_dim // 8)

        self.proj3 = nn.Conv3d(embed_dim //4 , embed_dim //8 ,(3,3,1),(1,1,1),(1,1,0))
        self.proj4 = nn.Conv3d(embed_dim //8 , embed_dim //4 ,(3,3,1), (1,1,1),(1,1,0))
        self.proj5 = nn.Conv3d(embed_dim //4 , embed_dim //2 ,(3,3,1),(1,1,1),(1,1,0))
        self.proj6 = nn.Conv3d(embed_dim//2,embed_dim ,(3,3,1),(1,1,1),(1,1,0))

    def forward(self, x):
        x = self.proj1(x)
        x = self.norm1(x)
        x = self.act1(x)

        x = self.proj2(x)
        x = self.norm2(x)
        x = self.act2(x)

        x = self.proj3(x)
        x = self.norm3(x)
        x = self.act3(x)

        x = self.proj4(x)
        x = self.norm2(x)

        x = self.proj5(x)
        x = self.norm1(x)
        x = self.act4(x)

        x = self.proj6(x)
        return x

class Subsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Subsample, self).__init__()
        self.conv = nn.Conv2d(in_channels , out_channels , kernel_size=3 , stride=2, padding= 1 )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4 , stride=2, padding=1)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.upsample(x)
        x = self.relu(x)
        return x

class Spe_squ(nn.Module):
    def __init__(self,in_channels,out_chennels):
        super(Spe_squ, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_chennels , kernel_size=1 ,stride=1, padding= 0)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.conv(x)
        x =self.relu(x)
        return x





class share_Weight(nn.Module):
    def __init__(self,in_,out_):
        super(share_Weight, self).__init__()
        hidden_ = in_ * 4
        self.conv_d1 = nn.Conv2d(in_,out_,kernel_size=5, stride=1 ,padding=2)
        self.conv_d2 = nn.Conv2d(in_,out_,kernel_size=7, stride=1 ,padding=3)
        self.conv_d3 = nn.Conv2d(in_,out_,kernel_size=9, stride=1 ,padding=4)

        self.conv = nn.Conv2d(out_*3 , out_, kernel_size=3 ,stride=1,padding=1)
        self.conv1d = nn.Linear(out_,out_)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.act3 = nn.ReLU()
        self.act4 = nn.ReLU()
        self.drop1 = nn.Dropout(0.1)
        self.drop2 = nn.Dropout(0.1)
        self.drop3 = nn.Dropout(0.1)
        self.drop = nn.Dropout(0.1)

    def forward(self,x):

        x1 = self.conv_d1(x)
        x1 = self.act1(x1)
        x1 = self.drop1(x1)

        x2 = self.conv_d2(x)
        x2 = self.act2(x2)
        x2 = self.drop2(x2)

        x3 = self.conv_d3(x)
        x3 = self.act3(x3)
        x3 = self.drop3(x3)

        temp = torch.cat([x1,x2,x3],dim=0)
        temp = self.conv(temp)
        temp = self.act4(temp)
        temp = temp.permute(1,2,0)
        temp = self.conv1d(temp)
        temp = self.drop(temp)

        return temp
class MMDMLP(nn.Module):
    def sliding_window(self,window_size,x):
        h , w, c = x.shape
        if h% window_size!=0 :
            x = torch.nn.functional.pad(x,(0,0,0,window_size,0,window_size))
            hh = h // window_size + 1
            h = hh * window_size
        else:
            hh = h // window_size

        if w % window_size != 0:
            ww = w // window_size + 1
            w = ww * window_size
        else :
            ww = w // window_size
        x = x[:h,:w,:]

        x = rearrange(x, '(h b0) (w b1) c -> (h w) b0 b1 c', h=hh, w=ww, b0=window_size, b1=window_size, c=c)

        return hh,ww,x


    def __init__(self, Patch, BAND, CLASSES_NUM, layers, embed_dims, segment_dim):
        super().__init__()
        global t_stride

        num_classes = CLASSES_NUM

        in_chans = 1
        layers = layers
        segment_dim = segment_dim
        mlp_ratios = 3
        embed_dims = embed_dims

        tmp = Patch
        qkv_bias = True
        C = int(embed_dims / segment_dim)

        drop_path_rate = 0.1
        norm_layer = nn.LayerNorm

        skip_lam = 1.0

        self.num_classes = num_classes

        self.patch_embed1 = PatchEmbed(in_chans=in_chans, embed_dim=embed_dims)
        self.patch_embed2 = PatchEmbed(in_chans=in_chans, embed_dim=embed_dims)
        self.patch_embed3 = PatchEmbed(in_chans=in_chans, embed_dim=embed_dims)
        self.patch_embed4 = PPatchEmbed()
        self.patch_embed5 = PPatchEmbed()
        self.patch_embed6 = PPatchEmbed()

        self.norm = norm_layer(linear_num)
        self.norm_hsi = norm_layer(linear_num)
        self.norm_lidar = norm_layer(linear_num)

        self.headx = nn.Linear(1088,linear_num)
        self.headxxx = nn.Linear(258,linear_num)
        self.headxx = nn.Linear(512,128)
        self.headxxxx = nn.Linear(64,128)

        self.head4 = nn.Linear(linear_num,num_classes)
        self.head4_lidar = nn.Linear(linear_num*2,linear_num)
        self.head4_fusion = nn.Linear(linear_num,num_classes)
        self.head4_xsi = nn.Linear(linear_num,num_classes)
        self.out1 = nn.Conv2d(linear_num*2,num_classes,1,1,0)
        self.out = nn.Linear(linear_num*3,num_classes)

        self.apply(self._init_weights)
        self.down_sample1 = Subsample(BAND,BAND)
        self.down_samplee = Subsample(BAND//4,BAND)
        self.down_samplee2 = Subsample(BAND//4,BAND)

        self.down_ss = Subsample(BAND,BAND)
        # Lidar 降采样

        self.up_sample = Upsample(BAND * 2 * embed_dims, BAND * 2 * embed_dims)
        self.up_sample = Upsample(BAND//2*embed_dims,BAND//2*embed_dims)
        self.up_sample2 = Upsample(BAND//2*embed_dims,BAND//2*embed_dims)
        self.up_sample3 = Upsample(BAND//2*embed_dims,BAND//2*embed_dims)

        self.conv1 = nn.Conv2d(linear_num,linear_num,1,1,0)
        self.conv2 = nn.Conv2d(linear_num,linear_num,1,1,0)
        self.conv3 = nn.Conv2d(linear_num,linear_num,1,1,0)
        self.conv4 = nn.Conv2d(linear_num,linear_num,1,1,0)
        self.conv5 = nn.Conv2d(linear_num,linear_num,1,1,0)
        self.conv6 = nn.Conv2d(linear_num,linear_num,1,1,0)




        self.conv7 = nn.Conv2d(linear_num,linear_num,1,1,0)
        self.conv8 = nn.Conv2d(linear_num,linear_num,3,1,1)
        self.conv9 = nn.Conv2d(linear_num,linear_num,1,1,0)
        self.conv10 = nn.Conv2d(linear_num,linear_num,3,1,1)
        #
        self.conv11 = nn.Conv2d(linear_num,linear_num,1,1,0)


        self.drop = nn.Dropout(0.1)


        #光谱带压缩
        self.squ_spe = Spe_squ(64,16)

        self.squ_spee = Spe_squ(64,16)
        self.squ_spees = Spe_squ(64,16)
        self.squ_spee2 = Spe_squ(64,16)

        self.norm1 = norm_layer(512)

        self.fusion = nn.Sequential(
                      nn.Conv2d(linear_num*2,linear_num,1,1,0),
                      nn.BatchNorm2d(linear_num),
                      nn.ReLU()
        )
        self.fusion_out = nn.Sequential(
                          nn.Conv2d(linear_num*2, linear_num,1,1,0),
                          nn.BatchNorm2d(linear_num),
                          nn.ReLU()
        )

        self.conv_hsi = nn.Linear(BAND,512)
        self.conv = nn.Linear(2,100)
        self.linear = nn.Linear(50,2)
        self.linearr1 = nn.Linear(2,64)
        self.linear1 = nn.Linear(BAND*embed_dims,BAND//2*embed_dims)
        self.linear2 = nn.Linear(BAND*embed_dims,BAND//2*embed_dims)
        self.linear3 = nn.Linear(BAND//2*embed_dims,embed_dims)
        self.linear4 = nn.Linear(BAND // 2 * embed_dims, embed_dims)
        self.linear5 = nn.Linear(BAND//8*embed_dims,2)
        self.linear6 = nn.Linear(BAND // 8 * embed_dims, 2)
         #3840 1280
        self.softmax = nn.Softmax(2)
        self.normx = nn.LayerNorm(linear_num)
        self.head_xsi = nn.Linear(linear_num,num_classes)

        self.share_weight = share_Weight(linear_num,linear_num)
        self.share_weight1 = share_Weight(linear_num,linear_num)
        self.share_weight2 = share_Weight(linear_num,linear_num)


        for name, p in self.named_parameters():
            if 't_fc.mlp_t.weight' in name:
                nn.init.constant_(p, 0)
            if 't_fc.mlp_t.bias' in name:
                nn.init.constant_(p, 0)
            if 't_fc.proj.weight' in name:
                nn.init.constant_(p, 1)
            if 't_fc.proj.bias' in name:
                nn.init.constant_(p, 0)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def get_pretrained_model(self, cfg):
        if cfg.MORPH.PRETRAIN_PATH:
            checkpoint = torch.load(cfg.MORPH.PRETRAIN_PATH, map_location='cpu')
            if self.num_classes != 1000:
                del checkpoint['head.weight']
                del checkpoint['head.bias']
            return checkpoint
        else:
            return None


    def forward_features(self, x,blockList):
        c = x
        c2 = x
        h,w,band = x.shape


        hh,ww,x = self.sliding_window(window_size1,x)
        x = x.view(x.shape[0],1,x.shape[1],x.shape[2],x.shape[3])
        # B,C,T,H,W -> B,T,H,W,C

        x = self.patch_embed1(x)
        x = x.permute(0, 4, 2, 3, 1)

        x1 = blockList(x,h,w,hh,ww,window_size1)


        # Window_Size2
        hh,ww,x = self.sliding_window(window_size2, c)
        x = x.view(x.shape[0],1,x.shape[1],x.shape[2],x.shape[3])

        x = self.patch_embed2(x)
        x = x.permute(0, 4, 2, 3, 1)
        x2 = blockList(x, h, w,hh,ww, window_size2)


        #Window_Size3
        hh,ww,x = self.sliding_window(window_size3, c2)
        x = x.view(x.shape[0], 1, x.shape[1], x.shape[2], x.shape[3])

        x = self.patch_embed3(x)

        x = x.permute(0, 4, 2, 3, 1)


        x3 = blockList(x, h, w,hh,ww, window_size3)

        output = x1 + x2 + x3


        return output

    def forward_features_lidar(self, x,blockList):
        c = x
        c2 = x
        h,w,band = x.shape


        hh,ww,x = self.sliding_window(window_size1,x)
        x = x.view(x.shape[0],1,x.shape[1],x.shape[2],x.shape[3])
        # B,C,T,H,W -> B,T,H,W,C

        x = self.patch_embed4(x)
        x = x.permute(0, 4, 2, 3, 1)

        x1 = blockList[3](x,h,w,hh,ww,window_size1)



        hh,ww,x = self.sliding_window(window_size2, c)
        x = x.view(x.shape[0],1,x.shape[1],x.shape[2],x.shape[3])

        x = self.patch_embed5(x)

        x = x.permute(0, 4, 2, 3, 1)

        x2 = blockList[4](x, h, w,hh,ww, window_size2)


        #Window_Size3
        hh,ww,x = self.sliding_window(window_size3, c2)
        x = x.view(x.shape[0], 1, x.shape[1], x.shape[2], x.shape[3])

        x = self.patch_embed6(x)

        x = x.permute(0, 4, 2, 3, 1)


        x3 = blockList[5](x, h, w,hh,ww, window_size3)

        output = x1 + x2 + x3

        return output

    def nonlocalfusion(self,x,y,z,conv1,conv2,conv3):
        temp = x.shape[2]
        rem = x
        x = x.view(x.shape[0] * x.shape[1],temp,1).permute(1,0,2)
        y = y.view(y.shape[0] * y.shape[1], temp,1).permute(1,0,2)
        z = z.view(z.shape[0] * z.shape[1], temp,1).permute(1,0,2)
        # rem = z
        x1 = conv1(x)
        y1 = conv2(y)
        x1 = x1.view(linear_num,1,-1)
        x1 = x1.permute(2,1,0)
        y1 = y1.view(linear_num,1,-1)
        y1 = y1.permute(2,1,0)

        z = conv3(z)
        z = z.view(linear_num,1,-1)
        z = z.permute(2,0,1)
        temp = self.softmax(torch.matmul(x1,z))
        temp = torch.matmul(temp,y1)
        out = temp.view(325,220,-1)

        out = out + rem
        return out

    def forward(self, hsi_input, lidar_input,blockList,epoch):
        h,w,c = hsi_input.shape
        x = hsi_input
        hsi_pure = hsi_input
        x = x.permute(2,0,1)
        x = self.squ_spees(x)
        x = x.permute(1,2,0)
        hsi_output = self.forward_features(x,blockList[0])


        hsi_input = hsi_input.permute(2,0,1)
        hsi_input1 = self.down_sample1(hsi_input)
        hsi_input1 = self.squ_spe(hsi_input1)
        hsi_input1 = hsi_input1.permute(1,2,0)
        hsi_output1 = self.forward_features(hsi_input1,blockList[1])

        hsi_input2 = hsi_input1.permute(2, 0, 1)
        hsi_input2 = self.down_samplee(hsi_input2)
        hsi_input2 = self.squ_spee(hsi_input2)
        hsi_input2 = hsi_input2.permute(1, 2, 0)
        hsi_output2 = self.forward_features(hsi_input2,blockList[2])


        xx = hsi_output2.permute(2,0,1)
        xx = self.up_sample2(xx)
        xx = xx.permute(1,2,0)
        xx = xx[:hsi_output1.shape[0],:hsi_output1.shape[1],:]
        hsi_out2 = hsi_output1 + xx

        hsi_out2 = hsi_out2.permute(2,0,1)
        hsi_out3 = self.up_sample3(hsi_out2)
        hsi_out3 = hsi_out3.permute(1,2,0)
        hsi_out3 = hsi_out3[:h,:w,:]
        hsi_output = torch.cat([hsi_out3,hsi_output,hsi_pure],dim=2)


        # Lidar

        lidar_output = self.forward_features_lidar(lidar_input,blockList)
        lidar_output = torch.cat([lidar_output,lidar_input],dim=2)

        hsi_output = self.headx(hsi_output)
        lidar_output = self.headxxx(lidar_output)

        hsi_output = self.share_weight(hsi_output)
        lidar_output = self.share_weight(lidar_output)



        #
        f1 = self.nonlocalfusion(lidar_output,lidar_output,hsi_output,self.conv1,self.conv2,self.conv3)
        f2 = self.nonlocalfusion(hsi_output,hsi_output,lidar_output,self.conv4,self.conv5,self.conv6)
        #
        f1 = f1.permute(2,0,1)
        f2 = f2.permute(2,0,1)
        y1 = F.max_pool2d(f1,3,1,1)
        y2 = F.max_pool2d(f2,3,1,1)
        #
        f1 =  y1 + f1
        f2 =  y2 + f2
        out = torch.cat([f1,f2],dim=0)
        out = self.out1(out)



        lidar_ = self.head4(f1)
        hsi_ = self.head4_xsi(f2)
        lidar_ = self.head4(lidar_)
        hsi_ = self.head4_xsi(hsi_)







        return out,hsi_,lidar_,_