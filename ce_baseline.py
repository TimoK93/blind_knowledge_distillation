# -*- coding:utf-8 -*-

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from data.datasets import input_dataset
from models import *
from utils import *
import argparse
import time
import os


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type = float, default = 0.1)
parser.add_argument('--val_ratio', type = float, default = 0.1)
parser.add_argument('--noise_type', type = str, help='clean, aggre, worst, rand1, rand2, rand3, clean100, noisy100', default='clean')
parser.add_argument('--noise_path', type = str, help='path of CIFAR-10_human.pt', default=None)
parser.add_argument('--dataset', type = str, help = ' cifar10 or cifar100', default = 'cifar10')
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--seed', type=int, default=0)  # we will test your code with 5 different seeds. The seeds are generated randomly and fixed for all participants.
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')



# Train the Model
def train(epoch, train_loader, model, optimizer):
    train_total=0
    train_correct=0
    model.train()
    for i, (images, labels, indexes) in enumerate(train_loader):

        batch_size = indexes.shape[0]
       
        images =images.to(args.device)
        labels =labels.to(args.device)
       
        # Forward + Backward + Optimize
        logits = model(images)

        prec, _ = accuracy(logits, labels, topk=(1, 5))
        train_total+=1
        train_correct+=prec
        loss = F.cross_entropy(logits, labels, reduce = True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % args.print_freq == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Training Accuracy: %.4F, Loss: %.4f'
                  %(epoch+1, args.n_epoch, i+1, len(train_dataset)//batch_size, prec, loss.data))


    train_acc=float(train_correct)/float(train_total)
    return train_acc

# Evaluate the Model
def evaluate(loader, model, save = False, best_acc = 0.0):
    model.eval()    # Change model to 'eval' mode.
    
    correct = 0
    total = 0
    for images, labels, _ in loader:
        images = Variable(images).to(args.device)
        logits = model(images)
        outputs = F.softmax(logits, dim=1)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred.cpu() == labels).sum()
    acc = 100*float(correct)/float(total)
    if save:
        if acc > best_acc:
            state = {'state_dict': model.state_dict(),
                     'epoch':epoch,
                     'acc':acc,
            }
            save_path= os.path.join('./', args.dataset + '_' + args.noise_type +'best.pth.tar')
            torch.save(state,save_path)
            best_acc = acc
            print(f'model saved to {save_path}!')

    return acc


##################################### main code ################################################
args = parser.parse_args()
# Seed
set_global_seeds(args.seed)
args.device = set_device()
time_start = time.time()
# Hyper Parameters
batch_size = 128
learning_rate = args.lr
noise_type_map = {'clean':'clean_label', 'worst': 'worse_label', 'aggre': 'aggre_label', 'rand1': 'random_label1', 'rand2': 'random_label2', 'rand3': 'random_label3', 'clean100': 'clean_label', 'noisy100': 'noisy_label'}
args.noise_type = noise_type_map[args.noise_type]
# load dataset
if args.noise_path is None:
    if args.dataset == 'cifar10':
        args.noise_path = './data/CIFAR-10_human.pt'
    elif args.dataset == 'cifar100':
        args.noise_path = './data/CIFAR-100_human.pt'
    else: 
        raise NameError(f'Undefined dataset {args.dataset}')


train_dataset, val_dataset, test_dataset, num_classes, num_training_samples = input_dataset(args.dataset,args.noise_type, args.noise_path, is_human = True, val_ratio = args.val_ratio)
# print('train_labels:', len(train_dataset.train_labels), train_dataset.train_labels[:10])
# load model
print('building model...')
model = ResNet34(num_classes)
print('building model done')
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                   batch_size = 128,
                                   num_workers=args.num_workers,
                                   shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                   batch_size = 128,
                                   num_workers=args.num_workers,
                                   shuffle=False)


test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                  batch_size = 64,
                                  num_workers=args.num_workers,
                                  shuffle=False)
alpha_plan = [0.1] * 50 + [0.01] * 50
model.to(args.device)



train_acc = 0.0
best_acc = 0.0
# training
for epoch in range(args.n_epoch):
# train models
    print(f'epoch {epoch}')
    adjust_learning_rate(optimizer, epoch, alpha_plan)
    # train models
    train_acc = train(epoch, train_loader, model, optimizer)
    print('train acc is ', train_acc)
    # evaluate models
    print('previous_best', best_acc)
    if args.val_ratio > 0.0:
        # save results
        val_acc = evaluate(loader=val_loader, model=model, save = True, best_acc = best_acc)
        if val_acc > best_acc:
            best_acc = val_acc
        print('validation acc is ', val_acc)
    test_acc = evaluate(loader=test_loader, model=model, save = False, best_acc = best_acc)
    print('test acc is ', test_acc)
    time_curr = time.time()
    time_elapsed = time_curr - time_start
    print(f'[Epoch {epoch}] Time elapsed {time_elapsed//3600:.0f}h {(time_elapsed%3600)//60:.0f}m {(time_elapsed%3600)%60:.0f}s', flush=True)



