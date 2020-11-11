from modules.encoder_cnn import Tag
from data_handlers.tag_loader import tag_loader
from pathlib import Path
from tqdm import tqdm
import numpy as np

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn as nn
import torch.nn.functional as F
from tools.ASL_loss import AsymmetricLossOptimized

class_num = 355
batch_size = 16
last_epoch = 0

device = torch.device('cuda')
model = Tag(class_num).to(device)
data_dir = Path(r'./data/data_splits')
train_data = tag_loader(data_dir, split='development',
                        batch_size=batch_size, class_num=class_num)
eval_data = tag_loader(data_dir, split='evaluation',
                       batch_size=batch_size, class_num=class_num)

learning_rate = 1e-3
optimizer = Adam(model.parameters(), lr=learning_rate,
                 betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)
scheduler = ExponentialLR(optimizer, gamma=0.98)
tag_loss = AsymmetricLossOptimized(gamma_neg=4, gamma_pos=1)


def tag_train(epoch):
    loss_list = []
    model.train()
    with tqdm(train_data, total=len(train_data)) as bar:
        for i, (feature, tag) in enumerate(bar):
            feature = feature.to(device)
            tag = tag.to(device)
            optimizer.zero_grad()
            out_tag = model(feature)
            loss = tag_loss(out_tag, tag)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            bar.set_description("epoch:{}, idx:{} loss:{:.6f}".format(epoch, i, np.mean(loss_list)))
    return np.mean(loss_list)


def tag_eval(epoch):
    loss_list = []
    model.eval()
    with torch.no_grad():
        for i, (feature, tag) in enumerate(eval_data):
            feature = feature.to(device)
            tag = tag.to(device)
            out_tag = model(feature)
            loss = tag_loss(out_tag, tag)
            loss_list.append(loss.item())
    mean_loss = np.mean(loss_list)
    print("epoch:{:d}--eval_loss:{:.6f}".format(epoch, mean_loss.item()))


if __name__ == "__main__":
    isTrain = True
    loadModel = False

    if loadModel:
        model.load_state_dict(torch.load(Path('./outputs/models/TagModel_4.pt')))

    if isTrain:
        for epoch in range(last_epoch + 1, last_epoch + 21):
            tag_train(epoch)
            scheduler.step(epoch)
            tag_eval(epoch)
            if epoch % 5 == 0:
                torch.save(model.state_dict(), './outputs/models/TagModel_{}.pt'.format(epoch))
