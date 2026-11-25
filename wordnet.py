import torch
from tools import builder
from utils.config import *
import numpy as np
import matplotlib.pyplot as plt
import itertools
import models.lorentz as L
# from models.meru_CrossModal import MERU_pretrained_model

_curv = torch.tensor(0.1000, device='cuda:0')

def save_feat():
    ckpt_path = '/home/ubuntu/CodeSpace/Hyperbolic_PC/experiments/base_meru/pretrain/meru/ckpt-best-93.pth'
    # ckpt_path = '/home/ubuntu/CodeSpace/Hyperbolic_PC/experiments/base_meru/pretrain/meru/ckpt-best.pth'
    map_location = {'cuda:%d' % 0: 'cuda:%d' % 0}
    config = cfg_from_yaml_file('cfgs/pretrain/base_meru.yaml')
    config.model['transformer_config'] = config.model.transformer_config
    base_model = builder.model_builder(config.model)
    text_encoder = base_model.encode_text
    text_part_encoder = base_model.encode_text_part
    if os.path.exists(ckpt_path):
        state_dict = torch.load(ckpt_path, map_location=map_location)
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
        base_model.load_state_dict(base_ckpt, strict=False)

    embeddings = []
    embeddings_part = []
    shapenet_dict = {"0":"airplane,aeroplane,plane",
                     "1":"ashcan,trash can,garbage can,wastebin,ash bin,ash-bin,ashbin,dustbin,trash barrel,trash bin",
                     # "2":"bag,traveling bag,travelling bag,grip,suitcase",
                     # "3":"basket,handbasket",
                     # "4":"bathtub,bathing tub,bath,tub",
                     # "5":"bed",
                     # "6":"bench",
                     # "7":"birdhouse",
                     # "8":"bookshelf",
                     # "9":"bottle",
                     # "10":"bowl",
                     # "11":"bus,autobus,coach,charabanc,double-decker,jitney,motorbus,motorcoach,omnibus,passenger vehi",
                     # "12":"cabinet",
                     # "13":"camera,photographic camera",
                     # "14":"can,tin,tin can",
                     # "15":"cap",
                     # "16":"car,auto,automobile,machine,motorcar",
                     # "17":"chair",
                     # "18":"clock",
                     # "19":"computer keyboard,keypad",
                     # "20":"dishwasher,dish washer,dishwashing machine",
                     # "21":"display,video display",
                     # "22":"earphone,earpiece,headphone,phone",
                     # "23":"faucet,spigot",
                     # "24":"file,file cabinet,filing cabinet",
                     # "25":"guitar",
                     # "26":"helmet",
                     # "27":"jar",
                     # "28":"knife",
                     # "29":"lamp",
                     # "30":"laptop,laptop computer",
                     # "31":"loudspeaker,speaker,speaker unit,loudspeaker system,speaker system",
                     # "32":"mailbox,letter box",
                     # "33":"microphone,mike",
                     # "34":"microwave,microwave oven",
                     # "35":"motorcycle,bike",
                     # "36":"mug",
                     # "37":"piano,pianoforte,forte-piano",
                     # "38":"pillow",
                     # "39":"pistol,handgun,side arm,shooting iron",
                     # "40":"pot,flowerpot","41":"printer,printing machine","42":"remote control,remote","43":"rifle","44":"rocket,projectile","45":"skateboard","46":"sofa,couch,lounge","47":"stove","48":"table","49":"telephone,phone,telephone set","50":"cellular telephone,cellular phone,cellphone,cell,mobile phone","51":"tower","52":"train,railroad train","53":"vessel,watercraft","54":"washer,automatic washer,washing machine"
                     }
    index = 0
    print(len(shapenet_dict))
    with torch.no_grad():
        for element in shapenet_dict:
            embeddings.append([])
            embeddings_part.append([])
            element = shapenet_dict[element].split(',')
            for text in element:
                embeddings[index].append(text_encoder(text).cpu())
                embeddings_part[index].append(text_part_encoder(text).cpu())
                # embeddings_part[index].append(text_part_encoder(text).cpu())
            index += 1
            print(element)

    torch.save(embeddings_part, "shapenet_dict_part.pt")
    torch.save(embeddings, "shapenet_dict.pt")


def tsne_2d(embeddings_value):
    from sklearn.manifold import TSNE
    # 创建t-SNE模型并进行降维
    tsne = TSNE(n_components=3, random_state=42)
    data_2d_tsne = tsne.fit_transform(embeddings_value)
    # 可视化
    plt.figure(figsize=(8, 6))
    # 定义一组颜色
    colors = plt.cm.tab10.colors
    label_colors = itertools.cycle(colors)
    for label in range(len(embeddings)):
        color_ = next(label_colors)
        plt.scatter(data_2d_tsne[label, 0], data_2d_tsne[label, 1], color=color_, alpha=0.3)
    plt.title('t-SNE caeter point 2D Visualization with Labels')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()


def UMAP_2d(embeddings_value):
    import umap
    colors = plt.cm.tab10.colors
    label_colors = iter(colors)
    label_colors = itertools.cycle(colors)
    umap_reducer = umap.UMAP(n_components=3, random_state=42)
    data_2d_umap = umap_reducer.fit_transform(embeddings_value)
    # 可视化
    plt.figure(figsize=(8, 6))
    for label in range(len(embeddings)):
        color_ = next(label_colors)
        plt.scatter(data_2d_umap[label, 0], data_2d_umap[label, 1], color=color_, alpha=0.3)
    plt.title('Umap caeter point 2D Visualization with Labels')
    plt.xlabel('Umap Component 1')
    plt.ylabel('Umap Component 2')
    plt.legend()
    plt.show()
    plt.show()


def tsne_3d(embeddings_value):
    from sklearn.manifold import TSNE
    # 创建t-SNE模型并进行降维
    tsne = TSNE(n_components=3, random_state=42)
    data_2d_tsne = tsne.fit_transform(embeddings_value)

    # 可视化
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    # 定义一组颜色
    colors = plt.cm.tab10.colors
    label_colors = iter(colors)
    label_colors = itertools.cycle(colors)
    for label in range(len(embeddings)):
        color_ = next(label_colors)
        ax.scatter(data_2d_tsne[label, 0], data_2d_tsne[label, 1], data_2d_tsne[label, 2], color=color_, alpha=1)

    ax.set_title('t-SNE 3D Visualization with Labels')
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.set_zlabel('t-SNE Component 3')
    plt.show()


def UMAP_3d(embeddings_value):
    import umap
    colors = plt.cm.tab10.colors
    label_colors = iter(colors)
    label_colors = itertools.cycle(colors)
    umap_reducer = umap.UMAP(n_components=3, random_state=42)
    data_2d_umap = umap_reducer.fit_transform(embeddings_value)
    # 可视化
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    for label in range(len(embeddings)):
        color_ = next(label_colors)
        ax.scatter(data_2d_umap[label, 0], data_2d_umap[label, 1], data_2d_umap[label, 2], color=color_, alpha=1)
    ax.set_title('UMAP 3D Visualization with Labels')
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.set_zlabel('t-SNE Component 3')
    plt.show()


def tsne_3d_scatters(embeddings_value):
    from sklearn.manifold import TSNE
    # 创建t-SNE模型并进行降维
    tsne = TSNE(n_components=3, random_state=42)
    data_2d_tsne = tsne.fit_transform(embeddings_value)

    # 可视化
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 定义一组颜色
    colors = plt.cm.tab10.colors
    label_colors = iter(colors)
    label_colors = itertools.cycle(colors)
    index = 0
    for label in range(len(embeddings)):
        color_ = next(label_colors)
        for j in range(len(embeddings[label])):
            if j % 2 == 0:
                ax.scatter(data_2d_tsne[index, 0], data_2d_tsne[index, 1], data_2d_tsne[index, 2], color=color_,
                           alpha=0.3)
                # plt.scatter(data_2d_tsne[index, 0], data_2d_tsne[index, 1], color = color_, alpha=0.3)
                # plt.scatter(data_2d_tsne[index, 0], data_2d_tsne[index, 1], label=f'Label {label}', color = color_, alpha=0.3)
            else:
                ax.scatter(data_2d_tsne[index, 0], data_2d_tsne[index, 1], data_2d_tsne[index, 2], color=color_,
                           alpha=1)
                # plt.scatter(data_2d_tsne[index, 0], data_2d_tsne[index, 1], color = color_, alpha=1)
                # plt.scatter(data_2d_tsne[index, 0], data_2d_tsne[index, 1], label=f'Label-part{label}', color = color_, alpha=1)

            index += 1
    # plt.title('t-SNE 2D Visualization with Labels')
    # plt.xlabel('t-SNE Component 1')
    # plt.ylabel('t-SNE Component 2')
    # plt.legend()
    ax.set_title('t-SNE 3D Visualization with Labels')
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.set_zlabel('t-SNE Component 3')
    plt.show()


def UMAP_3d_scatters(embeddings_value):
    import umap
    # 定义一组颜色
    colors = plt.cm.tab10.colors
    label_colors = iter(colors)
    label_colors = itertools.cycle(colors)
    # 创建UMAP模型并进行降维
    umap_reducer = umap.UMAP(n_components=3, random_state=42)
    data_2d_umap = umap_reducer.fit_transform(embeddings_value)
    # 可视化
    # 可视化
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    index = 0
    for label in range(len(embeddings)):
        color_ = next(label_colors)
        for j in range(len(embeddings[label])):
            if j % 2 == 0:
                ax.scatter(data_2d_umap[index, 0], data_2d_umap[index, 1], data_2d_umap[index, 2], color=color_,
                           alpha=0.3)
                # plt.scatter(data_2d_umap[index, 0], data_2d_umap[index, 1], data_2d_umap[index, 2], color = 'black', alpha=0.3)
                # plt.scatter(data_2d_umap[index, 0], data_2d_umap[index, 1], label=f'Label {label}', color = color_, alpha=0.3)
            else:
                ax.scatter(data_2d_umap[index, 0], data_2d_umap[index, 1], data_2d_umap[index, 2], color=color_,
                           alpha=1)
                # plt.scatter(data_2d_umap[index, 0], data_2d_umap[index, 1], data_2d_umap[index, 2], color = 'red', alpha=1)
                # plt.scatter(data_2d_umap[index, 0], data_2d_umap[index, 1], label=f'Label-part{label}', color = color_, alpha=1)
            index += 1
    # plt.title('UMAP 2D Visualization with Labels')
    # plt.xlabel('UMAP Component 1')
    # plt.ylabel('UMAP Component 2')
    # plt.legend()
    ax.set_title('UMAP 3D Visualization with Labels')
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.set_zlabel('t-SNE Component 3')
    plt.show()


def tsne_2d_scatters(embeddings_value):
    from sklearn.manifold import TSNE
    # 创建t-SNE模型并进行降维
    tsne = TSNE(n_components=3, random_state=42)
    data_2d_tsne = tsne.fit_transform(embeddings_value)

    # 可视化
    plt.figure(figsize=(8, 6))
    # 定义一组颜色
    colors = plt.cm.tab10.colors
    label_colors = iter(colors)
    label_colors = itertools.cycle(colors)
    index = 0
    for label in range(len(embeddings)):
        color_ = next(label_colors)
        for j in range(len(embeddings[label])):
            if j % 2 == 0:
                plt.scatter(data_2d_tsne[index, 0], data_2d_tsne[index, 1], color=color_, alpha=0.3)
                # plt.scatter(data_2d_tsne[index, 0], data_2d_tsne[index, 1], label=f'Label {label}', color = color_, alpha=0.3)
            else:
                plt.scatter(data_2d_tsne[index, 0], data_2d_tsne[index, 1], color=color_, alpha=1)
                # plt.scatter(data_2d_tsne[index, 0], data_2d_tsne[index, 1], label=f'Label-part{label}', color = color_, alpha=1)

            index += 1
    plt.title('t-SNE 2D Visualization with Labels')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.show()


def UMAP_2d_scatters(embeddings_value):
    import umap
    # 定义一组颜色
    colors = plt.cm.tab10.colors
    label_colors = iter(colors)
    label_colors = itertools.cycle(colors)
    # 创建UMAP模型并进行降维
    umap_reducer = umap.UMAP(n_components=3, random_state=42)
    data_2d_umap = umap_reducer.fit_transform(embeddings_value)
    # 可视化
    plt.figure(figsize=(8, 6))
    index = 0
    for label in range(len(embeddings)):
        color_ = next(label_colors)
        for j in range(len(embeddings[label])):
            if j % 2 == 0:
                plt.scatter(data_2d_umap[index, 0], data_2d_umap[index, 1], data_2d_umap[index, 2], color='black',
                            alpha=0.3)
                # plt.scatter(data_2d_umap[index, 0], data_2d_umap[index, 1], label=f'Label {label}', color = color_, alpha=0.3)
            else:
                plt.scatter(data_2d_umap[index, 0], data_2d_umap[index, 1], data_2d_umap[index, 2], color='red',
                            alpha=1)
                # plt.scatter(data_2d_umap[index, 0], data_2d_umap[index, 1], label=f'Label-part{label}', color = color_, alpha=1)
            index += 1
    plt.title('UMAP 2D Visualization with Labels')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.legend()
    plt.show()


save_feat()
embeddings = torch.load("shapenet_dict.pt")

for label in range(len(embeddings)):
    for j in range(len(embeddings[label])):
        if label==0 and j==0:
            embeddings_array = np.array(embeddings[0][0])
        else:
            embeddings_array = np.concatenate((embeddings_array, np.array(embeddings[label][j])), 0)
    len_label_his = len(embeddings[label])
    if label == 0:
        embeddings_center = L.refined_einstein_midpoint(torch.tensor(embeddings_array[-len_label_his: ]).cuda(), _curv).cpu().numpy()
    elif label > 0:
        embeddings_center = np.concatenate((embeddings_center,
                                            L.refined_einstein_midpoint(torch.tensor(embeddings_array[-len_label_his: ]).cuda(), _curv).cpu().numpy()), 0)

    # if len_label_his == 1:
    #     print(f'{torch.tensor(embeddings_array[-len_label_his: ]).cuda()-L.refined_einstein_midpoint(torch.tensor(embeddings_array[-len_label_his: ]).cuda(), _curv)}')
# embeddings_array = embeddings_array/np.linalg.norm(embeddings_array, axis=1, keepdims=True)
# embeddings_center = embeddings_center/np.linalg.norm(embeddings_center, axis=1, keepdims=True)


import umap
umap_reducer = umap.UMAP(n_components=3, random_state=42)
# data_2d_umap = umap_reducer.fit_transform(embeddings_array)
# data_2d_umap = np.concatenate((data_2d_umap, umap_reducer.fit_transform(embeddings_center)))
data_2d_umap = umap_reducer.fit_transform(np.concatenate((embeddings_array, embeddings_center), 0))

index = 0
label_his = 0
j_his = 0
index_his = 0
colors = plt.cm.tab10.colors
label_colors = itertools.cycle(colors)
print(f' embeddings_array.shape[0]:{embeddings_array.shape[0]:}--{len(embeddings)}')
index_his = 0
color_ = next(label_colors)

for data in data_2d_umap:
    if index < embeddings_array.shape[0]:
        print(f'==========={index} {embeddings_array.shape[0]}============')
        for label in range(label_his, len(embeddings)):
            for j in range(j_his, len(embeddings[label])):
                # print(f'search from: label_his-{label_his} j_his-{j_his}')
                # print(f'---label{label+1} of {len(embeddings)} | j{j+1} within [{len(embeddings[label])}]')
                index_his += 1
                if index_his == index + 1:
                    # if len(embeddings[label]) != 1:
                    plt.scatter(data[0], data[1], color=color_, alpha=0.5)
                    print(f'------found index {index} in label{label}-j{j}')
                    if j+1 < len(embeddings[label]):
                        label_his = label
                        j_his = j+1
                    else:
                        color_ = next(label_colors)
                        label_his = label + 1
                        j_his = 0
                    break
            if index_his == index + 1:
                break
    else:
        if index == embeddings_array.shape[0]:
            print(f'init_color--index{index}')
            label_colors = itertools.cycle(colors)
        color_ = next(label_colors)
        print(f'change--color_{color_}')
        plt.scatter(data[0], data[1], marker='^', color=color_, alpha=1, s=100)
    index = index + 1

plt.title('Umap 2D Visualization with Labels')
plt.xlabel('Umap Component 1')
plt.ylabel('Umap Component 2')
plt.legend()
plt.show()
