from torch import nn, optim
import logging
import Net
from ops import *
from Net import *
from sklearn.manifold import TSNE
CLASS_NUM = Net.CLASS_NUM
BAND = Net.BAND
LiDAR_BAND = 1
Epoch = 200
embeding_ = Net.EMBED_DIM
lidar_seg = 4
lidar_embeding_ = Net.Lidar_EMBED_DIM
seg = 4
Layers = 1
losses = []
test_acc = []
train_acc = []
per = 32
per2 = 2
gt = Net.gt

if not os.path.exists('logs/weights/'):
    os.makedirs('logs/weights/')

if torch.cuda.is_available():
    # 设置要使用的GPU设备编号（假设要使用第一个GPU）
    device = torch.device("cuda:3")
    # torch.cuda.manual_seed_all(326)
   # print("Using GPU:", torch.cuda.get_device_name(device))
else:
    device = torch.device("cpu")
    print("Using CPU")

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
    plt.savefig('/home/students/master/2022/liuy/PyCharm-Remote/learn/Hyperspectral_Image_Clustering/KD/'
                'ContrastiveLearning/SCEADNet/results/{}'
                .format(dataname) + '/tsne_{:.5f}'.format(acc) + '.png', format='png',
                transparent=True, pad_inches=0)
    # plt.show()
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
def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()

    torch.cuda.empty_cache()

    #device = torch.device("cuda:3")


    print("GPU Usage after emptying the cache")
    gpu_usage()
def Slice(Xh,per):
    if Xh.ndim == 3:
        rows,cols,_ = Xh.shape
    else:
        rows,cols = Xh.shape
    width = cols // per
    slices = []
    for i in range(per):
        start_col = i * width
        end_col = (i + 1) * width if i < per else cols
        sliced_array = Xh[:, start_col:end_col]
        slices.append(sliced_array)
    return slices
def Slices(Xh,per):
    if Xh.ndim == 4:
        siz,rows,cols,_ = Xh.shape
    else:
        siz,rows,cols = Xh.shape
    width = rows // per
    slices = []
    for t in range(siz):
        Xhh = Xh[t]
        for i in range(per):
            start_col = i * width
            end_col = (i + 1) * width if i < per else cols
            sliced_array = Xhh[start_col:end_col,:]
            slices.append(sliced_array)
    return slices
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

def data_Houston():
    # creat_all_houston18_1()
    Xh = np.load('./file/allg_Xh.npy')
    Xl = np.load('./file/allg_Xl.npy')
    Y = np.load('./file/allg_Y.npy')
    mask_ = np.load('./file/allg_mask.npy')
    temp_Xh = np.array(Slice(Xh,per))
    temp_Xl = np.array(Slice(Xl,per))
    temp_Y = np.array(Slice(Y,per))
    temp_mask = np.array(Slice(mask_,per))

    slice_Xh = Slices(temp_Xh,per2)
    slice_Xl = Slices(temp_Xl,per2)
    slice_Y = Slices(temp_Y,per2)
    slice_mask_ = Slices(temp_mask,per2)

    return slice_Xh,slice_Xl,slice_Y,slice_mask_

def adjust_lr(lr_init, lr_gamma, optimizer, epoch, step_index):
    # if epoch < 1:
    #     lr = 0.0001 * lr_initl
    # else:
    lr = lr_init * lr_gamma ** ((epoch - 1) // step_index)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr
def gen_log(model_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")

    log_file = model_path + '/log.txt'
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger
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
    setup_seed(91)
    model = SSMLP_allp(1,BAND,CLASS_NUM,Layers,embeding_,seg)
    model = model.to(device)


    # Xh ,Xl,Y,mask_ = data_Muufel()
    # Xh,Xl,Y,mask_ = data_trento()
    creat_all_houston18_1()
    S_h,S_l,S_Y,S_mask_ = data_Houston()

    S_h = np.array(S_h)
    S_h = S_h.astype(np.float32)
    S_h = torch.tensor(S_h)
    S_h = S_h.to(device)

    S_l = np.asarray(S_l)
    S_l = S_l.astype(np.float32)
    S_l = torch.tensor(S_l)
    S_l = S_l.unsqueeze(3)
    S_l = S_l.to(device)

    S_Y = np.asarray(S_Y)
    S_Y = S_Y.astype(np.int64)
    S_Y = torch.tensor(S_Y)
    S_Y = S_Y.to(device)

    S_mask_ = np.asarray(S_mask_)
    S_mask_ = S_mask_.astype(np.int64)
    S_mask_ = torch.tensor(S_mask_)
    S_mask_ = S_mask_.to(device)


    blockList = nn.ModuleList([])

    block1 = HsiBLock(1, BAND//gt, CLASS_NUM, 1, embeding_, seg, 1)
    block1.to(device)
    block2 = HsiBLock(1, BAND//gt, CLASS_NUM, 1, embeding_ , seg, 1)
    block2.to(device)
    block3 = HsiBLock(1, BAND//gt, CLASS_NUM, 1, embeding_, seg, 1)
    block3.to(device)
    blockList.append(block1)
    blockList.append(block2)
    blockList.append(block3)


    block4 = LidarBLock(1, LiDAR_BAND, CLASS_NUM, Layers,lidar_embeding_,lidar_seg,window_size1)#2860
    block4.to(device)
    block5 = LidarBLock(1, LiDAR_BAND, CLASS_NUM, Layers,lidar_embeding_,lidar_seg,window_size2)#2860
    block5.to(device)
    block6 = LidarBLock(1, LiDAR_BAND, CLASS_NUM, Layers,lidar_embeding_,lidar_seg,window_size3)#2860
    block6.to(device)


    blockList.append(block4)
    blockList.append(block5)
    blockList.append(block6)


    criterion = nn.CrossEntropyLoss().to(device)
    criterion2 = nn.MSELoss().to(device)

    optimizer = optim.Adam(model.parameters(),lr=0.002)
    epoch = 0


    best_acc = 0
    get = 0

    h = 1208
    w = 4768
    tmps = []
    alll = []
    for i in range(21):
        tmps.append(0)
        alll.append(0)
    a = 1
    b = 1
    c = 1
    temp_train = [[] for _ in range(per*per2)]
    temp_test = [[] for _ in range(per*per2)]
    Y_train_list = []
    Y_test_list = []
    for i in range(per*per2):
        Y_train_list.append(torch.Tensor())
        # Y_test_list.append(torch.Tensor())
    for k in range(per*per2):
        Y = S_Y[k]
        mask_ = S_mask_[k]
        for i in range(h // per2):
            for j in range(w // per):
                if(mask_[i][j] == 99):continue
                if(mask_[i][j] == 1):
                    temp_train[k].append(i * 1000000 + j)
                    if Y_train_list[k].numel() == 0:
                       Y_train_list[k] = Y[i][j].unsqueeze(0)
                    else:
                        Y_train_list[k] = torch.cat([Y_train_list[k],Y[i][j].unsqueeze(0)],dim=0)
                else:
                    if(mask_[i][j] == 2):
                        temp_test[k].append(i * 1000000 + j)
                    # if Y_test_list[k].numel() == 0:
                    #    Y_test_list[k] = Y[i][j].unsqueeze(0)
                    # else:
                    #     Y_test_list[k] = torch.cat([Y_test_list[k],Y[i][j].unsqueeze(0)],dim=0)
        if(k%10 == 0 ):
            print(k,"has complete.")
    randoms = np.zeros(per*per2)
    for i in range(per * per2):
        randoms[i] = i
        randoms = randoms.astype(int)
    np.random.shuffle(randoms)
    for epoch in range(Epoch):
        all_loss = 0
        les = 0
        # lr1 = adjust_lr(0.0005, 0.5, optimizer, epoch, 100)
        lr = adjust_lr(0.0002, 0.5, optimizer, epoch, 100)
        correct_fusion = 0
        test_correct_fusion = 0
        all = 0
        all2 = 0
        ans = 0
        np.random.shuffle(randoms)
        for batch in range(per*per2):
            batches = randoms[batch]
            Xh = S_h[batches]
            Xl = S_l[batches]
            output_train = torch.Tensor()
            lidar__ = torch.Tensor()
            hsi__ = torch.Tensor()
            flops, params = profile(model.to(device), inputs=(Xh, Xl, blockList, epoch))
            output,lidar_,hsi_,hsi_lidar_loss = model(Xh,Xl,blockList,epoch)
            lens = len(temp_train[batches])
            for temp in range(lens):
                x = temp_train[batches][temp] // 1000000
                y = temp_train[batches][temp] % 1000000
                if(output_train.numel()==0):
                    output_train = output[x][y].unsqueeze(0)
                    hsi__ = hsi_[x][y].unsqueeze(0)
                    lidar__ = lidar_[x][y].unsqueeze(0)
                    ans +=1
                else:
                    output_train = torch.cat([output_train,output[x][y].unsqueeze(0)],dim=0)
                    hsi__ = torch.cat([hsi__,hsi_[x][y].unsqueeze(0)],dim=0)
                    lidar__ = torch.cat([lidar__,lidar_[x][y].unsqueeze(0)],dim=0)
                    ans +=1
            Y_train = Y_train_list[batches]

            if(batch%50==0 or batch == 255):print("epoch:", epoch+1, "bacth:", batch, "has complete.","now size:",ans)
            # print("train set:",output_train.shape)

            if(Y_train.numel()==0):
                continue
            if(les==0):
                # lidar_loss = criterion(lidar__, Y_train)
                # hsi_loss = criterion(hsi__, Y_train)
                fusion_loss = criterion(output_train, Y_train)
                loss = fusion_loss
                les += 1
            else:
                # lidar_loss += criterion(lidar__, Y_train)
                # hsi_loss += criterion(hsi__, Y_train)
                fusion_loss += criterion(output_train, Y_train)
                loss = fusion_loss
                les += 1

            if(les == 1 or (batch == 255 and les != 0)):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                all_loss +=  loss.item()
                les = 0
        # print("epoch:",epoch +1 ,"hsi_loss:",hsi_loss.item(),"lidar_loss:",lidar_loss.item(),"fusion_loss:",fusion_loss.item())
        if(epoch % 1 == 0):
            with torch.no_grad():
                for kk in range(per*per2):
                    Xh = S_h[kk]
                    Xl = S_l[kk]
                    Y = S_Y[kk]
                    output, lidar_, hsi_, hsi_lidar_loss = model(Xh, Xl, blockList, epoch)
                    lens = len(temp_test[kk])
                    len2 = len(temp_train[kk])
                    for temp in range(lens):
                        x = temp_test[kk][temp] // 1000000
                        y = temp_test[kk][temp] % 1000000
                        test = F.softmax(output[x][y], dim=0).argmax()
                        if(test.item() == Y[x][y].item()):
                            test_correct_fusion +=1

                    for temp in range(len2):
                        x = temp_train[kk][temp] // 1000000
                        y = temp_train[kk][temp] % 1000000
                        train = F.softmax(output[x][y],dim=0).argmax()
                        if(train.item() == Y[x][y].item()):
                            correct_fusion +=1

                    all2 += lens
                    all += len2

            print("Train Acc:",100 * correct_fusion / all,  "Test Acc:",100*test_correct_fusion/all2,)
            print("-----------------------------------------------------------")
            test_acc.append(100 * test_correct_fusion/all2)
            train_acc.append(100 * correct_fusion / all)
            losses.append(all_loss)
            if  100*test_correct_fusion/all2  > best_acc:
                best_acc = 100*test_correct_fusion/ all2
                best_model_weights = model.state_dict()
                torch.save(best_model_weights, 'Houston_best_model.pth')
                with open('best.txt', 'w') as file:
                    file.write('Now best epoch:'+str(epoch)+'Test Acc:'+str(100 * test_correct_fusion / all2)+'Train Acc:'+str(100 * correct_fusion / all))
        correct = 0
    test_acc.append(100 * test_correct_fusion / all2)
    train_acc.append(100 * correct_fusion / all)
    losses.append(all_loss)
    final = 0
    final_test = 0
    #
    model.load_state_dict(torch.load('Houston_best_model.pth'))
    with torch.no_grad():
        tip= np.zeros(21)
        alls = np.zeros(21)
        for kk in range(per * per2):
            Xh = S_h[kk]
            Xl = S_l[kk]
            Y = S_Y[kk]
            mask_ = S_mask_[kk]
            output, lidar_, hsi_, hsi_lidar_loss = model(Xh, Xl, blockList, epoch)
            for i in range(h // per2):
                for j in range(w // per):
                    if (mask_[i][j] == 1):
                        continue
                    else:
                        if(mask_[i][j]!= 99):
                            final+=1
                            train = F.softmax(output[i][j], dim=0).argmax()
                            truth = Y[i][j].item()
                            alls[truth] +=1
                            if (train.item() == Y[i][j].item()):
                                tip[truth] +=1
                                final_test += 1
        print("Final Acc:",100*final_test / final)
        for i in range(20):
            print(tip[i],alls[i],tip[i]/alls[i],final_test,final)

    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(test_acc, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


    # torch.save(best_model_weights, 'best_model.pth')


    print("best acc:",best_acc)
def Draw_Classification_Map(acc, label, mapping, name: str, scale: float = 4.0, dpi: int = 400):
    # transformed_arr = np.array([mapping[element] for element in label.cpu().numpy()]).reshape(opt.args.height, opt.args.width)
    # label = transformed_arr
    if name is not None:
        colors = [  # 自定义颜色列表，每个类别对应一个颜色
            # (0, 0, 0),  # 类别0：黑色
            (58, 138, 71),  # 类别1：浅黄色
            (204, 180, 206),  # 类别2：蓝色
            (150, 84, 54),  # 类别3：橘红色
            (251, 193, 150),  # 类别4：深青色
            (137, 145, 200),  # 类别5：粉色
            (238, 45, 42),  # 类别6：蓝紫色
            (86, 132, 193),  # 类别7：天蓝色
            (128, 128, 128),  # 类别8：明绿色
            (128, 0, 0),  # 类别9：土黄色
            (128, 128, 0),  # 类别10：粉紫色
            (0, 128, 0),  # 类别11：浅蓝色
            (128, 0, 128),  # 类别12：深青色
            (0, 128, 128),  # 类别13：绿色
            (0, 0, 128),  # 类别14：棕色
            (255, 165, 0),  # 类别15：浅绿色
            (255, 215, 0),  # 类别16：明黄色
            (215,255,0),
            (150,84,54),
            (204,180,206),
            (58,138,71),
            (0, 0, 0),
            (0,0,0),
        ]
    # 创建一个空白的RGB图像，形状与label相同
    rgb_label = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)

    # 将颜色值赋给每个类别对应的像素
    for i, color in enumerate(colors):
        rgb_label[label == i] = color

    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.imshow(rgb_label)  # 显示彩色图像
    fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)

    foo_fig = plt.gcf()
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # foo_fig.savefig('/home/students/master/2022/liuy/PyCharm-Remote/learn/HSIC_student_Liuyi/DSCRLE_pytorch/src/L2GCC/'
    #                 'view_Label/{}'.format(opt.args.name) + '_{:.5f}'.format(acc) + '.png', format='png', transparent=True, dpi=dpi, pad_inches=0)
    plt.show()

def test():
    setup_seed(91)
    model = SSMLP_allp(1,BAND,CLASS_NUM,Layers,embeding_,seg)
    model = model.to(device)
    S_h,S_l,S_Y,S_mask_ = data_Houston()

    S_h = np.array(S_h)
    S_h = S_h.astype(np.float32)
    S_h = torch.tensor(S_h)
    S_h = S_h.to(device)

    S_l = np.asarray(S_l)
    S_l = S_l.astype(np.float32)
    S_l = torch.tensor(S_l)
    S_l = S_l.unsqueeze(3)
    S_l = S_l.to(device)

    S_Y = np.asarray(S_Y)
    S_Y = S_Y.astype(np.int64)
    S_Y = torch.tensor(S_Y)
    S_Y = S_Y.to(device)

    S_mask_ = np.asarray(S_mask_)
    S_mask_ = S_mask_.astype(np.int64)
    S_mask_ = torch.tensor(S_mask_)
    S_mask_ = S_mask_.to(device)


    blockList = nn.ModuleList([])

    block1 = HsiBLock(1, BAND//gt, CLASS_NUM, 1, embeding_, seg, 1)
    block1.to(device)
    block2 = HsiBLock(1, BAND//gt, CLASS_NUM, 1, embeding_ , seg, 1)
    block2.to(device)
    block3 = HsiBLock(1, BAND//gt, CLASS_NUM, 1, embeding_, seg, 1)
    block3.to(device)
    blockList.append(block1)
    blockList.append(block2)
    blockList.append(block3)


    block4 = LidarBLock(1, LiDAR_BAND, CLASS_NUM, Layers,lidar_embeding_,lidar_seg,window_size1)#2860
    block4.to(device)
    block5 = LidarBLock(1, LiDAR_BAND, CLASS_NUM, Layers,lidar_embeding_,lidar_seg,window_size2)#2860
    block5.to(device)
    block6 = LidarBLock(1, LiDAR_BAND, CLASS_NUM, Layers,lidar_embeding_,lidar_seg,window_size3)#2860
    block6.to(device)


    blockList.append(block4)
    blockList.append(block5)
    blockList.append(block6)


    criterion = nn.CrossEntropyLoss().to(device)
    criterion2 = nn.MSELoss().to(device)

    optimizer = optim.Adam(model.parameters(),lr=0.002)
    epoch = 0


    best_acc = 0
    get = 0

    h = 1208
    w = 4768
    tmps = []
    alll = []
    for i in range(21):
        tmps.append(0)
        alll.append(0)
    a = 1
    b = 1
    c = 1
    temp_train = [[] for _ in range(per*per2)]
    temp_test = [[] for _ in range(per*per2)]
    Y_train_list = []
    Y_test_list = []
    for i in range(per*per2):
        Y_train_list.append(torch.Tensor())
        # Y_test_list.append(torch.Tensor())
    for k in range(per*per2):
        Y = S_Y[k]
        mask_ = S_mask_[k]
        for i in range(h // per2):
            for j in range(w // per):
                if(mask_[i][j] == 99):continue
                if(mask_[i][j] == 1):
                    temp_train[k].append(i * 1000000 + j)
                    if Y_train_list[k].numel() == 0:
                       Y_train_list[k] = Y[i][j].unsqueeze(0)
                    else:
                        Y_train_list[k] = torch.cat([Y_train_list[k],Y[i][j].unsqueeze(0)],dim=0)
                else:
                    if(mask_[i][j] == 2):
                        temp_test[k].append(i * 1000000 + j)
                    # if Y_test_list[k].numel() == 0:
                    #    Y_test_list[k] = Y[i][j].unsqueeze(0)
                    # else:
                    #     Y_test_list[k] = torch.cat([Y_test_list[k],Y[i][j].unsqueeze(0)],dim=0)
        if(k%10 == 0 ):
            print(k,"has complete.")
    tmps = []
    alll = []
    alls = []
    tip = []

    for i in range(21):
        tmps.append(0)
        alll.append(0)
        alls.append(0)
        tip.append(0)


    a = 1
    b = 1
    c = 1
    final = 0
    final_test = 0
    model.load_state_dict(torch.load('Houston_best_model.pth'))
    Test = torch.Tensor()
    Ye = torch.Tensor()
    with torch.no_grad():
        for kk in range(per * per2):
            Xh = S_h[kk]
            Xl = S_l[kk]
            Y = S_Y[kk]
            output, lidar_, hsi_, hsi_lidar_loss = model(Xh, Xl, blockList, epoch)
            lens = len(temp_test[kk])
            len2 = len(temp_train[kk])
            for temp in range(lens):
                x = temp_test[kk][temp] // 1000000
                y = temp_test[kk][temp] % 1000000
                # test = F.softmax(output[x][y], dim=0).argmax()
                if( Test.numel() == 0) :
                    Test = output[x][y].unsqueeze(0)
                    Ye = Y[x][y].unsqueeze(0)
                else:
                    Test = torch.cat((Test,output[x][y].unsqueeze(0)),dim=0)
                    Ye = torch.cat((Ye,Y[x][y].unsqueeze(0)),dim=0)
    labelout = Test
    labelout = labelout.cpu().detach().numpy()
    Ye = Ye.cpu()
    tsne2d = TSNE(n_components=2, init='pca', random_state=0)
    X_tsne_2d = tsne2d.fit_transform(labelout)

    Draw_tsne(X_tsne_2d[:, 0:2], Ye, 90,'Trento')




    # temp_train = [[] for _ in range(per*per2)]
    # temp_test = [[] for _ in range(per*per2)]
    # Y_train_list = []
    # Y_test_list = []
    # model.load_state_dict(torch.load('Houston_best_model.pth'))
    # k1 = torch.Tensor()
    # k2 = torch.Tensor()
    # y1 = torch.Tensor()
    # y2 = torch.Tensor()
    #
    # with torch.no_grad():
    #     for kk in range(per):
    #         for j in range(per2):
    #             temp = kk*per2+j
    #             Xh = S_h[temp]
    #             Xl = S_l[temp]
    #             Y = S_Y[temp]
    #             Y = Y.cpu()
    #             output, lidar_, hsi_, hsi_lidar_loss = model(Xh, Xl, blockList, epoch)
    #             if(k2.numel()==0):
    #                 k2 = output
    #                 y2 = Y
    #             else:
    #                 k2 = torch.cat([k2,output],dim=0)
    #                 y2 = torch.cat([y2,Y],dim=0)
    #         if(k1.numel()==0):
    #             k1 = k2
    #             y1 = y2
    #         else:
    #             y1  = torch.cat([y1,y2],dim=1)
    #             k1 = torch.cat([k1,k2],dim=1)
    #         k2 = torch.Tensor()
    #         y2 = torch.Tensor()
    # labelout = k1.detach().softmax(dim=-1).argmax(dim=2, keepdim=True)
    # for i in range(1208):
    #     for j in range(4768):
    #         if(y1[i][j] == 99 or y1[i][j] == 20):
    #             labelout[i][j] = 21
    # # plt.plot(train_acc, label='Train Accuracy')
    # # plt.plot(test_acc, label='Test Accuracy')
    # # plt.xlabel('Epoch')
    # # plt.ylabel('Accuracy')
    # # plt.legend()
    # # plt.show()
    #
    # labelout = labelout.cpu()
    # labelout = labelout.squeeze()
    #
    # Draw_Classification_Map(90,labelout,None,'salinas')
    # Draw_Classification_Map(95,y1,None,'salinas')


    return output

def main():
    #train_two_branch()
    free_gpu_cache()
    # train_all_picture()
    test()
    #train_patch()

if __name__ == '__main__':
        main()
