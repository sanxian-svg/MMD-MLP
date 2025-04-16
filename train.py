from sklearn.metrics import confusion_matrix
from torch import nn, optim
import matplotlib
matplotlib.use('Agg')  # 必须在导入 pyplot 之前设置
from sklearn.manifold import TSNE
import Net
from ops import *
from Net import *
CLASS_NUM = 11
BAND = Net.BAND
Epoch = 200
embeding_ = 16
lidar_seg = 4
lidar_embeding_ = Net.Lidar_EMBED_DIM
seg = 4
Layers = 1
losses = []
test_acc = []
train_acc = []
l1_value = 0.5

if not os.path.exists('logs/weights/'):
    os.makedirs('logs/weights/')

if torch.cuda.is_available():
    # 设置要使用的GPU设备编号（假设要使用第一个GPU）
    device = torch.device("cuda:0")
    # torch.cuda.manual_seed_all(326)
   # print("Using GPU:", torch.cuda.get_device_name(device))
else:
    device = torch.device("cpu")
    print("Using CPU")
def setup_seed(seed):
    """
    setup random seed to fix the result
    Args:
        seed: random seed
    Returns: None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
def Draw_tsne(X, y, acc, dataname, title=None):
    """Plot an embedding X with the class label y colored by the domain d."""
    # X = X.cpu().numpy()
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Plot colors numbers
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    color = ["black", "yellow", "lightgreen", "indigo", "orange", "pink", "peru", "crimson", "aqua", "dodgerblue",
              "slategrey", "b", "red", "darkcyan", "grey", "olive", "green", "gold","magenta","cyan","silver","darkred","darkcyan","darkgray"]
    for i in range(X.shape[0]):
        if y[i] == 0:
            s0 = plt.scatter(X[i, 0], X[i, 1], marker='o', color=color[11])
        if y[i] == 1:
            s1 = plt.scatter(X[i, 0], X[i, 1], marker='o', color=color[1])
        if y[i] == 2:
            s2 = plt.scatter(X[i, 0], X[i, 1], marker='o', color=color[2])
        if y[i] == 3:
            s3 = plt.scatter(X[i, 0], X[i, 1], marker='o', color=color[3])
        if y[i] == 4:
            s4 = plt.scatter(X[i, 0], X[i, 1], marker='o', color=color[4])
        if y[i] == 5:
            s5 = plt.scatter(X[i, 0], X[i, 1], marker='o', color=color[5])
        if y[i] == 6:
            s6 = plt.scatter(X[i, 0], X[i, 1], marker='o', color=color[6])
        if y[i] == 7:
            s7 = plt.scatter(X[i, 0], X[i, 1], marker='o', color=color[7])
        if y[i] == 8:
            s8 = plt.scatter(X[i, 0], X[i, 1], marker='o', color=color[8])
        if y[i] == 9:
            s9 = plt.scatter(X[i, 0], X[i, 1], marker='o', color=color[9])
        if y[i] == 10:
            s10 = plt.scatter(X[i, 0], X[i, 1], marker='o', color=color[10])
        if y[i] == 11:
            s11 = plt.scatter(X[i, 0], X[i, 1], marker='o', color=color[11])
        if y[i] == 12:
            s12 = plt.scatter(X[i, 0], X[i, 1], marker='o', color=color[12])
        if y[i] == 13:
            s13 = plt.scatter(X[i, 0], X[i, 1], marker='o', color=color[13])
        if y[i] == 14:
            s14 = plt.scatter(X[i, 0], X[i, 1], marker='o', color=color[14])
        if y[i] == 15:
            s15 = plt.scatter(X[i, 0], X[i, 1], marker='o', color=color[15])
        if y[i] == 16:
            s16 = plt.scatter(X[i, 0], X[i, 1], marker='o', color=color[16])
        if y[i] == 17:
            s17 = plt.scatter(X[i, 0], X[i, 1], marker='o', color=color[17])
        if y[i] == 18:
            s18 = plt.scatter(X[i, 0], X[i, 1], marker='o', color=color[18])
        if y[i] == 19:
            s19 = plt.scatter(X[i, 0], X[i, 1], marker='o', color=color[19])
        if y[i] == 20:
            s20 = plt.scatter(X[i, 0], X[i, 1], marker='o', color=color[20])

    plt.xlabel('t-SNE:dimension 1')
    plt.ylabel('t-SNE:dimension 2')
    if title is not None:
        plt.title(title)
    if dataname == 'Muufl':
        plt.legend((s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10),
                   ("Trees","Mostly grass","Mixed ground surface","Dirt and sand","Road","Water","Building Shadow","Building","Sidewalk","Yellow curb","Cloth panels"),loc ='best')
    if dataname == 'Trento':
        plt.legend((s0,s1,s2,s3,s4,s5),
                   ("Apple trees","Buildings","Ground","Wood","Vineyard","Roads"),loc = 'best')
    if dataname == 'houston':
        plt.legend((s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15,s16,s17,s18,s19),
                   ("Healthy grass","Stressed grass","Artificial turf","Evergreen trees","Bare earth","Water","Residential buildings","Non-residential buildings","Roads","Sidewalks","Crosswalks","Major thoroughfares","Highways","Railways","Paved parking lots","Unpaved parking lots","Cars","Trains","Stadium seats"),loc = 'best')


    if dataname == 'indian':
        plt.legend((s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15),
                   ("Alfalfa", "Corn-notill", "Corn-mintill", "Corn", "Grass-pasture", "Grass-trees",
                    "Grass-pasture-mowed", "Hay-windrowed", "Oats", "Soybean-notill", "Soybean-mintill",
                    "Soybean-clean",
                    "Wheat",
                    "Woods", "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers"), loc='best')
    plt.savefig('/home/students/master/2023/huayy/code/Vimlp/tsne/' + 'muufl_multi+multi+multiD.png', format='png',
                transparent=True, pad_inches=0)
    plt.show()
def free_gpu_cache():
    print("Initial GPU Usage")
    # gpu_usage()

    torch.cuda.empty_cache()

    #device = torch.device("cuda:3")


    print("GPU Usage after emptying the cache")
    # gpu_usage()

def compute_multiclass_kappa(observed, expected):
    """
    计算多分类问题的Kappa系数

    参数:
    - observed: 观察到的分类结果 (一维数组)
    - expected: 期望的分类结果 (一维数组)

    返回:
    - kappa: 计算得到的Kappa系数
    """
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(observed, expected)

    # 计算总样本数 N
    N = conf_matrix.sum()

    # 计算观察到的准确率 Po
    Po = np.trace(conf_matrix) / N

    # 计算期望的准确率 Pe
    Pe = np.sum(np.sum(conf_matrix, axis=0) * np.sum(conf_matrix, axis=1)) / (N ** 2)

    # 计算Kappa系数
    kappa = (Po - Pe) / (1 - Pe)

    return kappa
def data_trento():
    creat_all_Trento()
    Xh = np.load('./file/allg_Xh.npy')
    Xl = np.load('./file/allg_Xl.npy')
    Y = np.load('./file/allg_Y.npy')
    mask_ = np.load('./file/allg_mask.npy')


    return Xh, Xl,Y,mask_
def data_Muufel():
    creat_all()
    Xh = np.load('./file/allg_Xh.npy')
    Xl = np.load('./file/allg_Xl.npy')
    Y = np.load('./file/allg_Y.npy')
    mask_ = np.load('./file/allg_mask.npy')
    return Xh, Xl, Y, mask_

def adjust_lr(lr_init, lr_gamma, optimizer, epoch, step_index):
    # if epoch < 1:
    #     lr = 0.0001 * lr_init
    # else:
    lr = lr_init * lr_gamma ** ((epoch - 1) // step_index)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr
def get_train():
    Y_train = np.load('./file/train_Y.npy')
    Y_train = Y_train.astype(np.int64)
    Y_train = torch.tensor(Y_train)
    # Y_train = F.one_hot(Y_train)

    # lidar训练数据
    Xl_train = np.load('./file/train_Xl.npy')
    Xl_train = torch.tensor(Xl_train)

    # lidar验证数据
    Xl_val = np.load('./file/val_Xl.npy')
    Y_val = np.load('./file/val_Y.npy')
    Xl_val = torch.tensor(Xl_val)
    Xl_train = Xl_train.permute(0, 3, 1, 2)
    Xl_val = Xl_val.permute(0, 3, 1, 2)
    Y_val = Y_val.astype(np.int64)
    Y_val = torch.tensor(Y_val)

    # hsi训练数据
    Xh_train = np.load('./file/train_Xh.npy  ')
    Xh_train = torch.tensor(Xh_train)

    Xh_val = np.load('./file/val_Xh.npy')
    Y_val = np.load('./file/val_Y.npy')

    Xh_val = torch.tensor(Xh_val)
    Xh_train = Xh_train.permute(0, 3, 1, 2)
    Xh_val = Xh_val.permute(0, 3, 1, 2)

    return Xh_train, Xl_train,Y_train,Xh_val,Xl_val,Y_val


def train_all_picture():
    setup_seed(71)
    model = MMDMLP(1,BAND,CLASS_NUM,Layers,embeding_,seg)
    model = model.to(device)


    Xh ,Xl,Y,mask_ = data_Muufel()
    # Xh,Xl,Y,mask_ = data_trento()
    Xh = Xh.astype(np.float32)
    Xh = torch.tensor(Xh)
    # Xh = Xh.to(device)


    Xl = Xl.astype(np.float32)
    Xl = torch.tensor(Xl)
    # Xl = Xl.to(device)

    Y = Y.astype(np.int64)
    Y = torch.tensor(Y)
    # Y = Y.to(device)

    mask_ = mask_.astype(np.int64)
    mask_ = torch.tensor(mask_)
    # mask_ = mask_.to(device)




    blockList = nn.ModuleList([])

    block1 = HsiBLock(1, BAND//4, CLASS_NUM, 1, embeding_, seg, 1)
    block1.to(device)
    block2 = HsiBLock(1, BAND//4, CLASS_NUM, 1, embeding_ , seg, 1)
    block2.to(device)
    block3 = HsiBLock(1, BAND//4, CLASS_NUM, 1, embeding_, seg, 1)
    block3.to(device)
    blockList.append(block1)
    blockList.append(block2)
    blockList.append(block3)


    block4 = LidarBLock(1, 2, CLASS_NUM, Layers,lidar_embeding_,lidar_seg,window_size1)#2860
    block4.to(device)
    block5 = LidarBLock(1, 2, CLASS_NUM, Layers,lidar_embeding_,lidar_seg,window_size2)#2860
    block5.to(device)
    block6 = LidarBLock(1, 2, CLASS_NUM, Layers,lidar_embeding_,lidar_seg,window_size3)#2860
    block6.to(device)


    blockList.append(block4)
    blockList.append(block5)
    blockList.append(block6)


    criterion = nn.CrossEntropyLoss().to(device)


    optimizer = optim.Adam(model.parameters(),lr=0.002)

    rem_label = np.zeros(1)
    rem_Y = np.zeros(1)

    best_acc = 0
    get = 0

    h,w = Y.shape
    tmps = []
    alll = []
    allls = []
    rem = []
    for i in range(15):
        tmps.append(0)
        alll.append(0)
        allls.append(0)
        rem.append(0)



    for epoch in range(Epoch):
        all_loss = 0
        output_train = torch.Tensor()
        Y_train = torch.Tensor()
        lidar__ = torch.Tensor()
        hsi__ = torch.Tensor()
        lr = adjust_lr(0.0005, 0.5, optimizer, epoch, 100)
        correct_fusion = 0
        test_correct_fusion = 0
        correct_hsi = 0
        test_correct_hsi = 0
        correct_lidar = 0
        test_correct_lidar = 0
        all = 0
        all2 = 0
        Xh = Xh.to(device)
        Xl = Xl.to(device)
        Y = Y.to(device)
        mask_ = mask_.to(device)
        # flops, params = profile(model.to(device), inputs=(Xh, Xl, blockList, epoch))
        output,lidar_,hsi_,hsi_lidar_loss = model(Xh,Xl,blockList,epoch)
        tmps = []
        for i in range(15):
            tmps.append(0)
        label_kappa = []
        Y_kappa = []
        for i in range(h):
            for j in range(w):
                if Y[i][j].item() == 99: continue
                if mask_[i][j].item() == 1 :
                    #print(Y[i][j])
                    tmps[Y[i][j].item()] += 1
                    if output_train.numel() == 0 :
                        output_train = output[i][j].unsqueeze(0)
                        lidar__ = lidar_[i][j].unsqueeze(0)
                        hsi__ = hsi_[i][j].unsqueeze(0)
                        Y_train = Y[i][j].unsqueeze(0)
                    else:
                        output_train = torch.cat((output_train,output[i][j].unsqueeze(0)),dim=0)
                        lidar__ = torch.cat((lidar__,lidar_[i][j].unsqueeze(0)),dim=0)
                        hsi__ = torch.cat((hsi__,hsi_[i][j].unsqueeze(0)),dim=0)
                        Y_train = torch.cat((Y_train,Y[i][j].unsqueeze(0)),dim = 0)
        if get == 0:
            print("train set:",output_train.shape)
            get += 1

        lidar_loss = criterion(lidar__,Y_train)
        hsi_loss = criterion(hsi__,Y_train)
        fusion_loss =  criterion(output_train,Y_train)

        loss = fusion_loss + hsi_loss + lidar_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        all_loss +=  loss.item()

        xout = output.detach()
        labelout = output.detach().softmax(dim=-1).argmax(dim=2, keepdim=True)
        hsiout = hsi_.detach().softmax(dim=-1).argmax(dim=2, keepdim=True)
        lidarout = lidar_.detach().softmax(dim=-1).argmax(dim=2, keepdim=True)
        for i in range(CLASS_NUM):
            tmps[i] =  0
            alll[i] = 0
            allls[i] = 0
        for a in range(h):
            for b in range(w):

                if Y[a][b].item() == 99:
                    continue
                else :
                    if mask_[a][b].item() == 1 :
                       all += 1
                       if labelout[a][b].item() == Y[a][b].item() :
                           correct_fusion += 1
                       if hsiout[a][b].item() == Y[a][b].item():
                           correct_hsi += 1
                       if lidarout[a][b].item() == Y[a][b].item() :
                           correct_lidar += 1
                    else:
                        label_kappa.append(labelout[a][b].item())
                        Y_kappa.append(Y[a][b].item())
                        x = Y[a][b].item()
                        allls[x] += 1
                        all2 += 1
                        if labelout[a][b].item() == Y[a][b].item():
                            test_correct_fusion += 1
                            alll[x] += 1
                        if hsiout[a][b].item() == Y[a][b].item():
                            test_correct_hsi += 1
                        if lidarout[a][b].item() == Y[a][b].item():
                            test_correct_lidar += 1
        a = correct_hsi / all
        b = correct_lidar / all
        c = correct_fusion / all

        aa = test_correct_hsi/ all2*100
        bb = test_correct_lidar / all2*100
        cc = test_correct_fusion / all2*100


        # print("epoc  h:", epoch + 1, ",", "a:",a*100,",b:",b*100,",c:",c*100)
        # print("epoch:", epoch + 1, ",", "a:", aa, ",b:", bb, ",c:", cc)
        # print("epoch:",epoch+1 , ",","hsi_loss:",hsi_loss.item(),",lidar_loss:",lidar_loss.item(),",fusion_loss:",fusion_loss.item())
        print("epoch:", epoch + 1, ",", "Loss:", all_loss,",", "Train Acc:", 100 * correct_fusion/all,"Test Acc:",100*test_correct_fusion/all2,"lr:",lr)
        print("-----------------------------------------------------------")
        if epoch+1 == 200:
            tsne2d = TSNE(n_components=2, init='pca', random_state=0)
            labelout = xout.cpu()
            labelout = labelout.reshape(-1,labelout.shape[2])
            label_x = labelout.numpy()

            Y = Y.cpu()
            Y = Y.reshape(-1)
            Y= Y.numpy()

            new_label = []
            new_x = []

            for label in range(11):
                indices = np.where(Y == label)[0]

                if len(indices)  >= 300:
                    selected_indices = np.random.choice(indices , 300, replace=False)
                else :
                    selected_indices = indices
                new_label.append(Y[selected_indices])
                new_x.append(label_x[selected_indices])
            new_label = np.concatenate(new_label,axis=0)
            new_x = np.concatenate(new_x,axis=0)
            new_x = torch.tensor(new_x)
            new_label = torch.tensor(new_label)



            X_tsne_2d = tsne2d.fit_transform(new_x)

            Draw_tsne(X_tsne_2d[:, 0:2], new_label, 90, 'Muufl_all')



        test_acc.append(100 * test_correct_fusion/all2)
        train_acc.append(100 * correct_fusion / all)
        losses.append(all_loss)
        if  100*test_correct_fusion/all2  > best_acc:
            best_acc = 100*test_correct_fusion/ all2
            best_model_weights = model.state_dict()
            for i in range(CLASS_NUM):
                rem[i] = alll[i] / allls[i] * 100
            rem_label = np.array(label_kappa)
            rem_Y = np.array(Y_kappa)
        correct = 0


    torch.save(best_model_weights, 'best_model.pth')

    model.load_state_dict(torch.load('best_model.pth'))
    print("best OA:",best_acc)
    ans = 0
    for i in range(CLASS_NUM):
        ans = ans + rem[i]
        print("class",i+1,":",rem[i])
    ans /= CLASS_NUM
    print("AA:",ans)
    print("Kappa:",compute_multiclass_kappa(rem_label,rem_Y))


def main():

    train_all_picture()

if __name__ == '__main__':
        main()
