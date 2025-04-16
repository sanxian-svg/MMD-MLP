# -*- coding: utf-8 -*-
import random

import numpy as np
import os
import cv2
import torch
import torch.nn.functional as F
from scipy.io import loadmat
from sklearn.decomposition import PCA

import scipy.io
#======== Dataset details========
NUM_CLASS = 6
SAVA_PATH = './file/'
BATCH_SIZE = 1
upscale = 2


hsi_dict = loadmat("Italy_hsi.mat")
lidar_dict = loadmat("Italy_lidar.mat")
gth_dict = loadmat("allgrd.mat")
predata = loadmat('muufl_gulfport_campus_1_hsi_220_label.mat')

lchn = 1 # number of lidar channels
hchn = 63 # number of hsi channels
torch.manual_seed(326)
if not os.path.exists(SAVA_PATH):
    os.mkdir(SAVA_PATH)
if torch.cuda.is_available():
    # 设置要使用的GPU设备编号（假设要使用第一个GPU）
    device = torch.device("cuda:3")
    # torch.cuda.manual_seed_all(326)
else:
    device = torch.device("cpu")
    print("Using CPU")
def Predict_Label2Img(predict_label):
    predict_img = torch.zeros([601,2385])
    num = predict_label.shape[0]

    for i in range(num):
        x = int(predict_label[i][1])
        y = int(predict_label[i][2])
        l = int(predict_label[i][3])
        predict_img[x][y] = l

    return predict_img
def load_torento():
    hsi_dict = loadmat("Italy_hsi.mat")
    lidar_dict = loadmat("Italy_lidar.mat")
    gth_dict = loadmat("allgrd.mat")

    hsi = hsi_dict['data']
    lidar = lidar_dict['data']
    gth = gth_dict['mask_test']
    return hsi,lidar,gth

def load_abusbrg():
    hsi_dict = loadmat("data_MS_LR.mat")
    lidar_dict = loadmat("data_SAR_HR.mat")
    gth_dict = loadmat("data_DSM.mat")

    hsi = hsi_dict['data_MS_LR']
    lidar = lidar_dict['data_SAR_HR']
    gth = gth_dict['data_DSM']
    return hsi,lidar,gth

def load_muufel():
    predata = loadmat("muufl_gulfport_campus_w_lidar_1.mat")
    hsi = predata['data']
    lidar = predata['']
def read_image(filename):
    img = tiff.imread(filename)
    img = np.asarray(img, dtype=np.float32)
    return img


def samele_wise_normalization(data):
    """
    normalize each sample to 0-1
    Input:
        sample
    Output:
        Normalized sample
    """
    if np.max(data) == np.min(data):
        return np.ones_like(data, dtype=np.float32) * 1e-6
    else:
        return 1.0 * (data - np.min(data)) / (np.max(data) - np.min(data))


def sample_wise_standardization(data):
    import math
    _mean = np.mean(data)
    _std = np.std(data)
    npixel = np.size(data) * 1.0
    min_stddev = 1.0 / math.sqrt(npixel)
    return (data - _mean) / max(_std, min_stddev)


def elastic_transform(image, alpha, sigma, random_state=None):
    import numpy as np
    from scipy.ndimage.interpolation import map_coordinates
    from scipy.ndimage.filters import gaussian_filter
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """

    if random_state is None:
        random_state = np.random.RandomState(None)

    if len(image.shape)==2:
        shape = image.shape
    else:
        shape=image.shape[:2]
        z=np.arange(image.shape[-1])

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))

    if len(image.shape)==2:
        return map_coordinates(image, indices, order=1).reshape(shape)
    else:
        for c in z:
            image[...,c]=map_coordinates(image[:,:,c],indices,order=1).reshape(shape)
        return image

def gth2mask(gth):
    # gth[gth>7]-=1
    # gth-=1
    new_gth = np.zeros(
        shape=(gth.shape[0], gth.shape[1], NUM_CLASS), dtype=np.int8)
    for c in range(NUM_CLASS):
        new_gth[gth == c, c] = 1
    return new_gth

def data_denerator(batch_size=50):

    hsi = hsi_dict['data']
    lidar = lidar_dict['data']
    gth = gth_dict['mask_test']
    hsi=samele_wise_normalization(hsi)
    lidar=samele_wise_normalization(lidar)
    gth = gth2mask(gth)
    frag = 0.20
    hm, wm = hsi.shape[0] - ksize, hsi.shape[1] - ksize
    Xh = []
    Xl = []
    Y = []
    index=0
    while True:
        idx = np.random.randint(hm)
        idy = np.random.randint(wm)
        tmph = hsi[idx:idx + ksize, idy:idy + ksize, :]
        tmpl = lidar[idx:idx + ksize, idy:idy + ksize]
        tmpy = gth[idx:idx + ksize, idy:idy + ksize,:]
        for c in range(1,NUM_CLASS):
            sm=np.sum(tmpy==c)
            if sm*1.0/(ksize**2)>frag:
                #增强数据
                if np.random.random() < 0.5:
                    tmph = np.flip(tmph, axis=0)
                    tmpl = np.flip(tmpl, axis=0)
                    tmpy = np.flip(tmpy, axis=0)
                if np.random.random() < 0.5:
                    tmph = np.flip(tmph, axis=1)
                    tmpl = np.flip(tmpl, axis=1)
                    tmpy = np.flip(tmpy, axis=1)
                #添加噪音
                if np.random.random() < 0.5:
                    noise = np.random.normal(0.0, 0.03, size=tmph.shape)
                    tmph += noise
                    noise = np.random.normal(0.0, 0.03, size=tmpl.shape)
                    tmpl += noise
                    Xh.append(tmph)
                    Xl.append(tmpl)
                    Y.append(tmpy)
                    index += 1
                    if index % batch_size == 0:
                        Xh = np.asarray(Xh, dtype=np.float32)
                        Xl = np.asarray(Xl, dtype=np.float32)
                        Xl=Xl[...,np.newaxis]
                        Y = np.asarray(Y, dtype=np.int8)
                        # yield([Xl, Xh], Y)
                        Xh = []
                        Xl = []
                        Y = []

def split_to_patches(hsi,lidar,icol):
    h, w, _ = hsi.shape
    ksize=2*r+1
    Xh = []
    Xl = []
    for i in range(0,h - ksize,ksize):
        Xh.append(hsi[i:i+ksize,icol:icol+ksize,:])
        Xl.append(lidar[i:i+ksize,icol:icol+ksize])
    Xh=np.asarray(Xh,dtype=np.float32)
    Xl=np.asarray(Xl,dtype=np.float32)
    Xl=Xl[...,np.newaxis]
    return Xl,Xh


def creat_patches(batch_size=50,validation=False):
    print()
    hsi = hsi_dict['data']
    lidar = lidar_dict['data']
    gth = gth_dict['mask_test']
    hsi = np.pad(hsi, ((r, r), (r, r), (0, 0)), 'symmetric')
    lidar = np.pad(lidar, ((r, r), (r, r) ), 'symmetric')
    gth = np.pad(gth, ((r, r), (r, r)), 'constant', constant_values=(0,0))
    lidar=samele_wise_normalization(lidar)
    hsi=samele_wise_normalization(hsi)
    lidar-=np.mean(lidar)
    hsi-=np.mean(hsi)
    print(np.amax(gth))
    Xh=[]
    Xl=[]
    Y=[]
    count=0
    per = 0.11
    idx,idy=np.where(gth!=0)
    ID=np.random.permutation(len(idx))
    idx=idx[ID]
    idy=idy[ID]
    if not validation:
        idx=idx[:int(per*len(idx))]
        idy=idy[:int(per*len(idy))]
    else:
        idx=idx[int(per*len(idx)):]
        idy=idy[int(per*len(idy)):]
    while True:
        for i in range(len(idx)):
            tmph=hsi[idx[i]-r:idx[i]+r+1,idy[i]-r:idy[i]+r+1,:]
            tmpl=lidar[idx[i]-r:idx[i]+r+1,idy[i]-r:idy[i]+r+1]
            tmpy=gth[idx[i],idy[i]]-1
            # tmph=sample_wise_standardization(tmph)
            # tmpl=sample_wise_standardization(tmpl)
            if not validation:
                if np.random.random() < 0.5:
                    tmph = np.flip(tmph, axis=0)
                    tmpl = np.flip(tmpl, axis=0)
                if np.random.random() < 0.5:
                    tmph = np.flip(tmph, axis=1)
                    tmpl = np.flip(tmpl, axis=1)
                if np.random.random()<0.5:
                    k=np.random.randint(4)
                    tmph=np.rot90(tmph,k=k)
                    tmpl=np.rot90(tmpl,k=k)
            Xh.append(tmph)
            Xl.append(tmpl)
            Y.append(tmpy)
            count+=1
            if count % batch_size == 0:
                Xh = np.asarray(Xh, dtype=np.float32)
                Xl = np.asarray(Xl, dtype=np.float32)
                # Xc = np.reshape(Xh[:, r, r, :], [-1, 1, hchn])
                Xl=Xl[...,np.newaxis]
                Y = np.asarray(Y, dtype=np.int8)
                #独热编码
                #Y = to_categorical(Y, NUM_CLASS)
                Y = F.one_hot(Y,NUM_CLASS)
                Y = Y.numpy()
                yield([Xl,Xh], Y)
                Xh = []
                Xl = []
                Y = []

def down_sampling_hsi(hsi,scale=2):
    hsi = cv2.GaussianBlur(hsi, (3, 3), 0)
    hsi = cv2.resize(cv2.resize(hsi,
                                (hsi.shape[1] // scale, hsi.shape[0] // scale),
                                interpolation=cv2.INTER_CUBIC),
                     (hsi.shape[1], hsi.shape[0]),
                     interpolation=cv2.INTER_CUBIC)
    return hsi
def applyPCA(X, numComponents):
    """
    apply PCA to the image to reduce dimensionality
  """
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX
def creat_all_houston():
    dict = loadmat('houston_data.mat')
    hsi_dict = loadmat('data_HS_LR.mat')
    # hsi = hsi_dict['data_HS_LR']
    hsi = dict['hsi']
    lidar = dict['lidar']
    lidar = sample_wise_standardization(lidar)
    hsi = sample_wise_standardization(hsi)
    # hsi = applyPCA(hsi,64)
    gth_dict1 = dict['train']
    gth_dict2 = dict['test']

    cnt = 0
    select_ = np.zeros([349,1905])
    gth = np.zeros([349,1905])
    flag = np.zeros([349,1905])
    all1 =  0
    all2 = 0
    tip = np.zeros([15])
    for i in range(349):
        for j in range(1905):
            gth_dict1[i][j] -= 1
            gth_dict2[i][j] -= 1
            if(gth_dict1[i][j]>=0 and gth_dict1[i][j] < 15):
                select_[i][j] = 1
                gth[i][j] = gth_dict1[i][j]
                all1 += 1
                continue
            if(gth_dict2[i][j]>=0 and gth_dict2[i][j] < 15):
                select_[i][j] = 2
                gth[i][j] = gth_dict2[i][j]
                all2 += 1
            if(gth_dict2[i][j]==255 and gth_dict1[i][j]==255):
                gth[i][j] = 99
                select_[i][j] = 99

    print('train hsi data shape:{},train lidar data shape:{}'.format(hsi.shape, lidar.shape))
    np.save(os.path.join(SAVA_PATH, 'allg_Xh.npy'), hsi)
    np.save(os.path.join(SAVA_PATH, 'allg_Xl.npy'), lidar)
    np.save(os.path.join(SAVA_PATH, 'allg_Y.npy'), gth)
    np.save(os.path.join(SAVA_PATH, 'allg_mask.npy'), select_)

def creat_all_houston18_1():
    dict = loadmat('houston_hsi.mat')
    # hsi = hsi_dict['data_HS_LR']
    # dict = loadmat('GRSS2018.mat')
    hsi = dict['houston_hsi']
    lidar_dict = loadmat('houston_lidar.mat')
    lidar = lidar_dict['houston_lidar']
    lidar = sample_wise_standardization(lidar)
    hsi = sample_wise_standardization(hsi)
    pad_width = ((0, 6), (0, 0), (0, 0))
    pad_width1 = ((0, 6), (0, 0))
    hsi = np.pad(hsi,pad_width,mode='reflect')
    lidar = np.pad(lidar,pad_width1,mode='reflect')

    # hsi = applyPCA(hsi,24)
    gth_dict = loadmat('houston_gt.mat')
    gth = gth_dict['houston_gt']
    gth = np.pad(gth,pad_width1,mode='reflect')
    st = []
    tip = np.zeros([21])
    select_ = np.zeros([1208, 4768])
    h = 1202
    w = 4768
    for i in range(21):
        st.append(0)
    kk = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],[]]
    for i in range(0, 1208):
        for j in range(0, 4768):
            if(i>=1202):
                select_[i][j] = 99
                continue
            st[gth[i][j]]+=1
            if gth[i][j] == 0:
                gth[i][j] = 100
            gth[i][j] -=1
            # cnt += 1
            # gth[i][j] = gth[i][j] - 1
            if gth[i][j] != 99:
                kk[gth[i][j]].append(i * 100000 + j)
            else:
                select_[i][j] = 99

    for i in range(20):
        random.shuffle(kk[i])
    flag = 0
    ts = [500,500,68,500,500,451,26,500,800,800,800,800,500,500,500,500,14,500,500,500]
    temp = 0
    for i in range(20):
        tick = min(300+ts[i],len(kk[i]))
        for j in range(tick):
            if kk[i][j] == -1:
                flag = 1
                break
            if(j<=ts[i]):
                select_[kk[i][j] // 100000][kk[i][j] % 100000] = 1
                temp +=1
            else:
                select_[kk[i][j]//100000][kk[i][j] % 100000] = 2
                # temp2 += 0
            if j == tick or flag == 1:
                break

    select_ = torch.tensor(select_)
    select_ = select_.reshape(1208, 4768)
    select_ = select_.numpy()

    print('train hsi data shape:{},train lidar data shape:{}'.format(hsi.shape, lidar.shape))
    hsi = hsi[:,:,:48]
    np.save(os.path.join(SAVA_PATH, 'allg_Xh.npy'), hsi)
    np.save(os.path.join(SAVA_PATH, 'allg_Xl.npy'), lidar)
    np.save(os.path.join(SAVA_PATH, 'allg_Y.npy'), gth)
    np.save(os.path.join(SAVA_PATH, 'allg_mask.npy'), select_)


def creat_all_Trento():

    hsi = hsi_dict['data']
    lidar = lidar_dict['data']
    gth = gth_dict['mask_test']
    lidar = sample_wise_standardization(lidar)
    hsi = sample_wise_standardization(hsi)
    cnt = 0
    select_ = np.zeros([166, 600])
    id = np.zeros((166 * 600))
    kk = [[], [], [], [], [], [], [], [], [], [], []]
    st = []
    for i in range(10):
        st.append(0)
    for i in range(0, 166):
        for j in range(0, 600):
            st[gth[i][j]]+=1
            if gth[i][j] == 0:
                gth[i][j] = 100
            gth[i][j] -=1
            cnt += 1
            # gth[i][j] = gth[i][j] - 1
            if gth[i][j] != 99:
                kk[gth[i][j]].append(i * 100000 + j)

    for i in range(6):
        random.shuffle(kk[i])
    flag = 0
    for i in range(6):
        for j in range(151):
            if kk[i][j] == -1:
                flag = 1
                break
            if j == 150 or flag == 1:
                break
            select_[kk[i][j] // 100000][kk[i][j] % 100000] = 1

    select_ = torch.tensor(select_)
    select_ = select_.reshape(166, 600)
    all = 0
    for i in range(166):
        for j in range(600):
            if(select_[i][j].item()==1):
                all +=1
    select_ = select_.numpy()
    Xh = []
    Xl = []
    Y = []

    print('train hsi data shape:{},train lidar data shape:{}'.format(hsi.shape, lidar.shape))
    np.save(os.path.join(SAVA_PATH, 'allg_Xh.npy'), hsi)
    np.save(os.path.join(SAVA_PATH, 'allg_Xl.npy'), lidar)
    np.save(os.path.join(SAVA_PATH, 'allg_Y.npy'), gth)
    np.save(os.path.join(SAVA_PATH, 'allg_mask.npy'), select_)



def creat_all():
    tmp = predata['hsi']
    hsi = predata['hsi']['Data'][0][0]
    h = hsi.shape[0]
    w = hsi.shape[1]
    c = hsi.shape[2]
    lidar = predata['hsi']['Lidar'][0][0][0][0]['z'][0][0]
    gth =  predata['hsi']['sceneLabels'][0][0]['labels'][0][0]
    # hsi = hsi.reshape(hsi.shape[0]*hsi.shape[1],hsi.shape[2])
    # pca = PCA(n_components=16)
    # hsi_p = pca.fit_transform(hsi)
    # hsi_p = hsi_p.reshape(h,w,16)
    # hsi = hsi_p
    per = 1
    lidar = sample_wise_standardization(lidar)
    hsi = sample_wise_standardization(hsi)
    cnt = 0
    select_ = np.zeros([325,220])
    id = np.zeros((325*220))
    kk = [[],[],[],[],[],[],[],[],[],[],[]]
    for i in range(0, 325):
        for j in range(0, 220):
            if gth[i][j]==-1:
                gth[i][j] = 100
            cnt += 1
            gth[i][j] = gth[i][j] - 1
            if gth[i][j]!= 99 :
                kk[gth[i][j]].append(i*100000+j)

    for i in range(11):
       random.shuffle(kk[i])
    flag = 0
    xs = int(100*per)
    ll = []
    for i in range(11):
        ll.append(xs)
        if i == 3 or i==4 or i==5 or i ==9 or i==10 :
            ll[i] = 5
    for i in range(11):
        for j in range(ll[i]):
            #print("label:",i,"x:",kk[i][j]//100000,"y:",kk[i][j]%100000)
            if kk[i][j] == -1:
                flag = 1
                break
            if j == xs or flag == 1 :
                break
            select_[kk[i][j]//100000][kk[i][j]%100000] = 1




    # for i in range(len(id_train)):
    #     select_[index[i]] = 1
    # for i in range(len(id_train),len(index)):
    #     select_[index[i]] = 0

    select_ = torch.tensor(select_)
    select_ = select_.reshape(325,220)
    select_ = select_.numpy()
    Xh = []
    Xl = []
    Y = []
    Train_test = []

    # lidar = sample_wise_standardization(lidar)
    # hsi = sample_wise_standardization(hsi)

    # pca = PCA(n_components=16)
    # hsi = pca.fit_transform(hsi)
    #
    Xh = np.asarray(Xh, dtype=np.float32)
    Xl = np.asarray(Xl, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.int8)

    print('train hsi data shape:{},train lidar data shape:{}'.format(hsi.shape,lidar.shape))
    np.save(os.path.join(SAVA_PATH, 'allg_Xh.npy'), hsi)
    np.save(os.path.join(SAVA_PATH, 'allg_Xl.npy'), lidar)
    np.save(os.path.join(SAVA_PATH, 'allg_Y.npy'), gth)
    np.save(os.path.join(SAVA_PATH,'allg_mask.npy'),select_)










def creat_train(validation=False,r=[1,2,3]):
    tmp = predata['hsi']
    hsi = predata['hsi']['Data'][0][0]
    lidar = predata['hsi']['Lidar'][0][0][0][0]['z'][0][0]
    gth =  predata['hsi']['sceneLabels'][0][0]['labels'][0][0]

    a = r[2]

    hsi = np.pad(hsi, ((a, a), (a, a), (0, 0)), 'symmetric')
    if len(lidar.shape) == 2:
        lidar = np.pad(lidar, ((a, a), (  a, a)), 'symmetric')
    if len(lidar.shape) == 3:
        lidar = np.pad(lidar, ((a, a), (a, a), (0, 0)), 'symmetric')
    gth = np.pad(gth, ((a, a), (a, a)), 'constant', constant_values=(0, 0))
    per = 0.01
    
    if lchn == 2:
        hsi = down_sampling_hsi(hsi,upscale)

    lidar = sample_wise_standardization(lidar)
    hsi = sample_wise_standardization(hsi)
    # hsi=whiten(hsi)

    Xh1 = []
    Xl1 = []
    Y = []
    Xh2 = []
    Xl2 = []
    Xh3 = []
    Xl3 = []
    for c in range(1, NUM_CLASS):
        idx, idy = np.where(gth == c)
       # print("idx:",idx)
       # print("idy:",idy)
        #print("------------------------------")
        if not validation:
            idx = idx[:int(per * len(idx))]
            idy = idy[:int(per * len(idy))]
        else:
            idx = idx[int(per * len(idx)):]
            idy = idy[int(per * len(idy)):]
        random_number = np.random.randint(1, 100000)
        np.random.seed(random_number)
        ID = np.random.permutation(len(idx))
        idx = idx[ID]
        idy = idy[ID]
        for i in range(len(idx)):
            tmph1 = hsi[idx[i] - r[0]:idx[i] + r[0] + 1, idy[i] - r[0]:idy[i] + r[0] + 1,:]
            tmph2 = hsi[idx[i] - r[1]:idx[i] + r[1] + 1, idy[i] - r[1]:idy[i] + r[1] + 1,:]
            tmph3 = hsi[idx[i] - r[2]:idx[i] + r[2] + 1, idy[i] - r[2]:idy[i] + r[2] + 1, :]

            #添加了:
            tmpl1 = lidar[idx[i] - r[0]:idx[i] + r[0] + 1, idy[i] - r[0]:idy[i] + r[0] + 1]
            tmpl2 = lidar[idx[i] - r[1]:idx[i] + r[1] + 1, idy[i] - r[1]:idy[i] + r[1] + 1]
            tmpl3 = lidar[idx[i] - r[2]:idx[i] + r[2] + 1, idy[i] - r[2]:idy[i] + r[2] + 1]
            tmpy = gth[idx[i], idy[i]] - 1

            Xh1.append(tmph1)
            Xl1.append(tmpl1)


            Xh2.append(tmph2)
            Xl2.append(tmpl2)


            Xh3.append(tmph3)
            Xl3.append(tmpl3)



            Y.append(tmpy)

    index = np.random.permutation(len(Xh1))
    Xh1 = np.asarray(Xh1, dtype=np.float32)
    Xl1 = np.asarray(Xl1, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.int8)
    Xh1 = Xh1[index, ...]

    Xh2 = np.asarray(Xh2, dtype=np.float32)
    Xl2 = np.asarray(Xl2, dtype=np.float32)
    Xh2 = Xh1[index, ...]

    Xh3 = np.asarray(Xh3, dtype=np.float32)
    Xl3 = np.asarray(Xl3, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.int8)
    Xh3 = Xh3[index, ...]

    if len(Xl1.shape)==3:
        Xl1 = Xl1[index, ..., np.newaxis]
    elif len(Xl1.shape)==4:
        Xl1 = Xl1[index, ...]
    Y = Y[index]
    print('train hsi data shape:{},train lidar data shape:{}'.format(Xh1.shape,Xl1.shape))
    if not validation:

        np.save(os.path.join(SAVA_PATH, 'train_Xh1.npy'), Xh1)
        np.save(os.path.join(SAVA_PATH, 'train_Xl1.npy'), Xl1)

        np.save(os.path.join(SAVA_PATH, 'train_Xh2.npy'), Xh2)
        np.save(os.path.join(SAVA_PATH, 'train_Xl2.npy'), Xl2)

        np.save(os.path.join(SAVA_PATH, 'train_Xh3.npy'), Xh3)
        np.save(os.path.join(SAVA_PATH, 'train_Xl3.npy'), Xl3)

        np.save(os.path.join(SAVA_PATH, 'train_Y.npy'), Y)
    else:

        np.save(os.path.join(SAVA_PATH, 'val_Xh1.npy'), Xh1)
        np.save(os.path.join(SAVA_PATH, 'val_Xl1.npy'), Xl1)

        np.save(os.path.join(SAVA_PATH, 'val_Xh2.npy'), Xh2)
        np.save(os.path.join(SAVA_PATH, 'val_Xl2.npy'), Xl2)

        np.save(os.path.join(SAVA_PATH, 'val_Xh3.npy'), Xh3)
        np.save(os.path.join(SAVA_PATH, 'val_Xl3.npy'), Xl3)

        np.save(os.path.join(SAVA_PATH, 'val_Y.npy'), Y)


def make_cTest():
    hsi = read_image(os.path.join(PATH, HsiName))
    lidar = read_image(os.path.join(PATH, LiDarName))
    gth = tiff.imread(os.path.join(PATH, gth_test))

    hsi = np.pad(hsi, ((r, r), (r, r), (0, 0)), 'symmetric')
    if len(lidar.shape) == 2:
        lidar = np.pad(lidar, ((r, r), (r, r)), 'symmetric')
    if len(lidar.shape) == 3:
        lidar = np.pad(lidar, ((r, r), (r, r), (0, 0)), 'symmetric')
    gth = np.pad(gth, ((r, r), (r, r)), 'constant', constant_values=(0, 0))

    hsi = down_sampling_hsi(hsi)

    lidar = sample_wise_standardization(lidar)
    hsi = sample_wise_standardization(hsi)
    # hsi=whiten(hsi)
    idx, idy = np.where(gth != 0)
    ID = np.random.permutation(len(idx))
    Xh = []
    Xl = []
    for i in range(len(idx)):
        tmph = hsi[idx[ID[i]] - r:idx[ID[i]] + r +
                   1, idy[ID[i]] - r:idy[ID[i]] + r + 1, :]
        tmpl = lidar[idx[ID[i]] - r:idx[ID[i]] +
                     r + 1, idy[ID[i]] - r:idy[ID[i]] + r + 1]
        tmpy = gth[idx[ID[i]], idy[ID[i]]] - 1
        # tmph=sample_wise_standardization(tmph)
        # tmpl=sample_wise_standardization(tmpl)
        Xh.append(tmph)
        Xl.append(tmpl)
    Xh = np.asarray(Xh, dtype=np.float32)
    Xl = np.asarray(Xl, dtype=np.float32)
    index = np.concatenate(
        (idx[..., np.newaxis], idy[..., np.newaxis]), axis=1)
    np.save(os.path.join(SAVA_PATH, 'hsi.npy'), Xh)
    np.save(os.path.join(SAVA_PATH, 'lidar.npy'), Xl)
    np.save(os.path.join(SAVA_PATH, 'index.npy'), [idx[ID] - r, idy[ID] - r])
    if len(Xl.shape) == 3:
        Xl=Xl[ ..., np.newaxis]
    return Xl, Xh

