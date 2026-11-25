import os
import torch
import numpy as np
from torch.utils.data import Dataset

def xyz_normalize(point):
    '''
    point:[[x1,y1,z1],[x2,y2,z2],...,[xn,yn,zn]]
    计算采样点与中心点之间的距离,并且将距离归一化
    '''
    centroid=np.mean(point,axis=0)
    point=point-centroid
    dist=np.max(np.sqrt(np.sum(point**2,axis=1)))
    point=point/dist
    return point

class ModelNetDataLoader(Dataset):
    def __init__(self,root,npoint=1024,split="train"):
        self.root=root      # 数据集的根目录
        self.npoint=npoint  # 每个点云数据的采样点个数

        # 记录40个类别名称的文件
        self.catfile=os.path.join(self.root,"modelnet40_shape_names.txt")
        # 获取40个类别名称,["airplane",...,"xbox"]
        self.cat=[line.strip() for line in open(self.catfile)]
        # 获取40个类别名称及其对应索引,{"airplane":0,...,"xbox":39}
        self.classes=dict(zip(self.cat,range(len(self.cat))))

        shape_ids={}
        # 获取训练集每个点云数据的文件名,["airplane_0001",...,"xbox_0103"]
        shape_ids["train"]=[line.strip() for line in open(os.path.join(self.root,"modelnet40_train.txt"))]
        # 获取验证集每个点云数据的文件名,["airplane_0627",...,"xbox_0123"]
        shape_ids["test"]=[line.strip() for line in open(os.path.join(self.root,"modelnet40_test.txt"))]

        assert (split=="train" or split=="test")
        # 获取训练集(验证集)每个点云数据的类别名称,["airplane",...,"xbox"]
        shape_names=["".join(x.split("_")[0:-1]) for x in shape_ids[split]]

        # 获取训练集(验证集),[(类别,路径),...,(类别,路径)]
        self.datapath=[(shape_names[i],os.path.join(self.root,shape_names[i],shape_ids[split][i])+".txt") for i in range(len(shape_ids[split]))]

        # 训练集:9843; 验证集:2468
        print("The size of %s data is %d"%(split,len(self.datapath)))

    def __len__(self):
        # 返回训练集(验证集)大小
        return len(self.datapath)

    def __getitem__(self,index):

        # 获取一个点云数据,(类别,路径)
        fn=self.datapath[index]

        # 获取点云数据的类别所对应的索引,如"airplane":0
        cls=self.classes[self.datapath[index][0]]
        cls=np.array([cls]).astype(np.int32)

        # 根据点云数据的路径加载点云数据,[[x1,y1,z1,nx1,ny1,nz1],...]
        point_set=np.loadtxt(fn[1],delimiter=",").astype(np.float32)
        # 前1024个点作为采样点
        point_set=point_set[0:self.npoint,:]
        # 对采样点的(x,y,z)做标准化处理
        point_set[:,0:3]=xyz_normalize(point_set[:,0:3])

        return point_set,cls    # (1024*6,[id])

DATA_PATH="/home/ubuntu/CodeSpace/data/ModelNet/modelnet40_normal_resampled"

train_dataset=ModelNetDataLoader(root=DATA_PATH,npoint=1024,split="train")
test_dataset=ModelNetDataLoader(root=DATA_PATH,npoint=1024,split="test")
# 创建用于加载训练集的Dataloader
extra_train_dataloader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=8,shuffle=True,num_workers=0)
extra_test_dataloader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=8,shuffle=True,num_workers=0)
