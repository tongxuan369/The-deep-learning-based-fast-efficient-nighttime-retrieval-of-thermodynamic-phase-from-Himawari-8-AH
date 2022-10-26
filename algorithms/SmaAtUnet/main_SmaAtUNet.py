#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.abspath("./models"))
import torch
import torch.nn as nn
import torch.nn.functional as F
import code
import numpy as np
from torch.autograd import Variable
import time
import torch.utils.data as Data
from math import sqrt
import os
from SmaAt_UNet import SmaAt_UNet
import torch.optim as optim
import ssl
import random
from makedata import MyDataset
import torch.utils.data.distributed as tdd
import torch.distributed as distributed
import torch.multiprocessing
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
def get_backend_file():
    file_dir = os.path.dirname(__file__)
    file_name = os.path.basename(__file__) + '.bin'
    backend_file = os.path.realpath(os.path.join(file_dir, file_name))
    init_method = 'file://%s' % backend_file
    return init_method

def train(gpu_id, gpu_count):
    t0_train = time.time()
    BATCH_SIZE = 18
    Epoch = 300
    weight_decay = 0.01
    inchannel = 16
    outchannel = 4
    lrate = 0.001
    echo_interval = 10
    data_train_dir = './phaseQA'
    random_array_dir = './random_array.npy'
    path_data_maxmin = './xmaxmin_whole.npy'
    is_master = gpu_id == 0
    default_device = torch.device('cuda', gpu_id)
    default_type = torch.float32
    init_method = get_backend_file()
    distributed.init_process_group("nccl", init_method=init_method, rank=gpu_id, world_size=gpu_count)
    print(
        "*** Running at %s | Batch Size: %d | GPU count: %d | Total batch size: %d " % (
            default_device, BATCH_SIZE, gpu_count, BATCH_SIZE * gpu_count))
    ssl._create_default_https_context = ssl._create_unverified_context
    net = SmaAt_UNet(n_channels=inchannel, n_classes=outchannel)
    net.to(default_device, dtype=torch.float32)
    net = DistributedDataParallel(net, device_ids=[gpu_id], output_device=gpu_id)
    optimizer = ZeroRedundancyOptimizer(
        net.parameters(),
        optimizer_class=torch.optim.AdamW,
        lr=lrate,
        weight_decay=weight_decay
    )
    loss_fn = nn.CrossEntropyLoss()
    loss_fn.to(default_device, dtype=default_type)
    dealDataset_train = MyDataset(data_train_dir, path_data_maxmin, random_array_dir,_train=True)
    sampler = tdd.DistributedSampler(dealDataset_train, shuffle=True)
    train_loader = Data.DataLoader(dataset=dealDataset_train, sampler=sampler, batch_size=BATCH_SIZE, shuffle=False,
                                   drop_last=True, num_workers=10, prefetch_factor=2, pin_memory=True)
    train_step = 0
    loss_epoch = []
    tt = time.time()
    lossbest = 99999
    for epoch in range(Epoch):
        sampler.set_epoch(epoch)
        train_loss = []
        batchtime_begin = time.time()
        for inputs, labels in train_loader:
            inputs = inputs.to(device=default_device, dtype=torch.float32)
            labels = labels.to(device=default_device, dtype=torch.long)
            if train_step == 0:
                print("input:shape", inputs.shape, labels.shape)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.detach().cpu().numpy())
            train_step += 1
            if is_master and train_step % echo_interval == 0:
                tpb = (time.time() - tt) / echo_interval
                tt = time.time()
                loss_avg = np.average(train_loss)
                print("[Epoch %5d | Batch %5d] loss : %5.5f | Time/Batch: %5.5fs" % (
                    epoch, (train_step - 1), loss_avg, tpb))
        loss_ = np.average(train_loss)
        if lossbest > loss_:
            net_file = os.path.join(os.getcwd(), 'model.pth')
            distributed.barrier()
            optimizer.consolidate_state_dict()
            if is_master:
                torch.save({
                    'model_state_dict': net.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, net_file)
                print("*** save state to file %s" % net_file)
            distributed.barrier()
            distributed.destroy_process_group()
            lossbest = loss_
        loss_epoch.append(loss_)
        batchtime_end = time.time()
        print('Batch time=', batchtime_end - batchtime_begin)
    t1_train = time.time()
    np.save('loss.npy', np.array(loss_epoch))
    print('Training time=', t1_train - t0_train)

def test():
    t0_test = time.time()
    inchannel = 18
    outchannel = 4
    path_data_maxmin = './xmaxmin_whole.npy'
    random_array_dir = './random_array_test.npy'
    default_device = torch.device("cuda:0") if torch.cuda.is_available() else ("cpu")
    ssl._create_default_https_context = ssl._create_unverified_context  # 在80服务器上不加这个 下面ResUnet预训练回报错
    net = SmaAt_UNet(n_channels=inchannel, n_classes=outchannel)
    net.to(default_device, dtype=torch.float32)
    model_file = os.path.join(os.getcwd(), 'model.pth')
    pk = torch.load(model_file, map_location=default_device)
    net.load_state_dict(pk['model_state_dict'])
    net.eval()
    data_test_dir = './phaseQA'
    dealDataset_test = MyDataset(data_test_dir, path_data_maxmin, random_array_dir,_train=False)
    test_loader = Data.DataLoader(dataset=dealDataset_test, batch_size=1, shuffle=False, drop_last=False)  # =====测试集总长度
    test_np = []
    real_np = []
    for i, (xx, yy) in enumerate(test_loader):
        inputs_test = xx.to(device=default_device, dtype=torch.float32)
        outputs_test = net(inputs_test)
        predict = outputs_test.detach().cpu().numpy()
        test_np.append(predict)
        real_np.append(yy.detach().cpu().numpy())
    np.save('./predict.npy', np.array(test_np))
    np.save('./real.npy', np.array(real_np))
    t1_test = time.time()
    print('Test time=', t1_test - t0_test)

def train_model():
    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        print("require gpu count >= 1, current is %s" % gpu_count)
        exit(1)
    torch.multiprocessing.spawn(train, args=(gpu_count,), nprocs=gpu_count, join=True)

if __name__ == "__main__":
    setup_seed(20)
    t0 = time.time()
    train_model()
    test()
    t1 = time.time()
    print('All work is done, time=', t1 - t0)
    code.interact(local=locals())
