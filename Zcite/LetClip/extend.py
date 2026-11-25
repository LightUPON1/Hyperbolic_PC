import torch
import copy
import os
import torch.nn as nn
from options import args_parser

def schimit(P):
    n_components = P.shape[0]
    coeff = torch.eye(n_components).cuda()
    for i in range(n_components):
        if i > 0:
            tmp = torch.mm(P[:i, :], P[i].reshape(-1, 1))
            P[i] -= torch.mm(P[:i, :].T, tmp).reshape(-1)
            coeff[i] -= torch.mm(coeff[:i, :].T, tmp).reshape(-1)
        tmp = torch.norm(P[i])
        P[i] /= tmp
        coeff[i] /= tmp
    return P
def get_local_weights_vec(model):
    """
    Return model parameters as a vector
    """
    vec = []
    for key in model.keys():
        vec.append(model[key].clone().detach().reshape(-1))
    return torch.cat(vec, 0)

def get_model_grad_vec( model):
    # Return the model grad as a vector
    vec = []
    for name, param in model.named_parameters():
        vec.append(param.grad.detach().reshape(-1))
    return torch.cat(vec, 0)
def reverse_P_SGD(model, optimizer, grad, P):
    # P_plus_BFGS algorithm
    gk = torch.mm(P, grad.reshape(-1, 1))
    grad_proj = torch.mm(P.transpose(0, 1), gk)
    idx = 0
    for name, param in model.named_parameters():
        arr_shape = param.data.shape
        size = 1
        for i in range(len(list(arr_shape))):
            size *= arr_shape[i]
        param.grad.data = param.grad.data - grad_proj.reshape(arr_shape)
        idx += size
    optimizer.step()

k = 10
NUM_HYPERPARAM = k
NUM_TASKS = 10
local_ep = 200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
args = args_parser()

# TODO: model
model = model.to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.5)

local_weights = []
local_losses = []

# TODO: trainloader

theta_0 = None
local_weights.append(copy.deepcopy(model.state_dict()))

for i in range(NUM_HYPERPARAM+1):
    for iter in range(local_ep):
        batch_loss = []
        for batch_idx, (images, labels) in enumerate(trainloader[0]):
            images, labels = images.to(device), labels.to(device)
            model.zero_grad()
            log_probs = model(images)
            loss = criterion(log_probs, labels)
            loss.backward()
            optimizer.step()
    if theta_0 == None:
        theta_0 = get_local_weights_vec(copy.deepcopy(model.state_dict())).reshape(1, -1)
    local_weights.append(copy.deepcopy(model.state_dict()))

# model_paths = [os.path.join(args.model_location, f'model_{i}.pt') for i in range(NUM_HYPERPARAM)]
# for j, model_path in enumerate(model_paths):
theta_set = (get_local_weights_vec(local_weights[0]).reshape(1, -1))
for i in range(1, len(local_weights)):
    theta_set = torch.cat((theta_set, get_local_weights_vec(local_weights[i]).reshape(1, -1)))
delta_theta = theta_set - theta_0

# P_theta = schimit(theta_set)
P_delta_theta =  schimit(delta_theta)


for task_idx in range(NUM_TASKS):
    if task_idx > 0:
        local_weights = []
        for i in range(NUM_HYPERPARAM):
            for iter in range(local_ep):
                batch_loss = []
                for batch_idx, (images, labels) in enumerate(trainloader[task_idx]):
                    images, labels = images.to(device), labels.to(device)
                    model.zero_grad()
                    log_probs = model(images)
                    loss = criterion(log_probs, labels)
                    loss.backward()
                    mode = 1
                    if mode == 1:
                        # TODO: plan1
                        gk = get_model_grad_vec(model)
                        tmp = torch.mm(gk, gk.reshape(-1, 1))
                        gk -= torch.mm(gk.T, tmp).reshape(-1)
                        tmp = torch.norm(gk)
                        gk /= tmp
                        optimizer.step()

                    if mode == 2:
                        # TODO: plan2
                        gk = get_model_grad_vec(model)
                        reverse_P_SGD(model=model, optimizer=optimizer, grad=gk, P=P_delta_theta)

                    if mode == 3:
                        # TODO: plan2
                        gk = get_model_grad_vec(model)
                        reverse_P_SGD(model=model, optimizer=optimizer, grad=gk, P=P_theta)

            local_weights.append(copy.deepcopy(model.state_dict()))