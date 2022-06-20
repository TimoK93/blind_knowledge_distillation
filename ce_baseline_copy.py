# -*- coding:utf-8 -*-
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from data.datasets import input_dataset
from models import *
from utils import *
import argparse
import time
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.02)
parser.add_argument('--val_ratio', type=float, default=0.0)
parser.add_argument('--noise_type', type=str, help='clean, aggre, worst, rand1, rand2, rand3, clean100, noisy100',
                    default='clean')
parser.add_argument('--noise_path', type=str, help='path of CIFAR-10_human.pt', default=None)
parser.add_argument('--dataset', type=str, help=' cifar10 or cifar100', default='cifar10')
parser.add_argument('--n_epoch', type=int, default=300)
parser.add_argument('--seed', type=int,
                    default=0)  # we will test your code with 5 different seeds. The seeds are generated randomly and fixed for all participants.
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--batch_size', default=300, type=int, help='train batch-size')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD optimizer')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay for SGD optimizer')
parser.add_argument('--alpha_final', type=float, default=0.6, help='final value for alpha in bootstrapped loss')
parser.add_argument('--trusted_to_0', type=bool, default=False,
                    help='set alpha for trusted labels to 0 in bootstrapped target')
parser.add_argument('--untrusted_to_1', type=bool, default=False,
                    help='set alpha for untrusted labels to 1 in bootstrapped target')
parser.add_argument('--incremental_label_correction', type=bool, default=False, help='N.A.')
parser.add_argument('--label_smoothing', type=float, default=0.0, help='label smoothing for naive loss')
parser.add_argument('--training', type=bool, default=True)
parser.add_argument('--stop_after_overfitting', type=bool, default=False)
parser.add_argument('--divide_dataset', type=bool, default=True)
parser.add_argument('--split_dataset', type=str, default='otsu_thresh', help='otsu_thresh, otsu_up_mean')
parser.add_argument('--correct_labels', type=bool, default=True)
parser.add_argument('--clean_test_freq', type=int, default=1, help='how often the current model performance on clean data will be displayed')
parser.add_argument('--save_model_freq', type=int, default=2, help='freq for saving state of models. will be auto saved at end of training')
args = parser.parse_args()


def my_train(current_epoch, teacher, optimizer_teacher, dataloader, student, optimizer_student, group_1, group_2, group_3, group_4):
    teacher.train()
    student.train()
    prob_teacher_container, prob_student_container, prob_agree_container = \
        torch.zeros((len(dataloader.dataset))).to(args.device), \
        torch.zeros((len(dataloader.dataset))).to(args.device),\
        torch.zeros((len(dataloader.dataset))).to(args.device)
    mean_max_student_prediction = 0
    number_of_samples = 0
    for batch_idx, (img, target, index) in enumerate(dataloader):
        img = img.to(args.device)
        target = target.to(args.device)

        # Load modified labels
        img, target = img.to(args.device), target.to(args.device)
        pred_teacher = teacher(img)
        pred_student = student(img)

        one_hot = F.one_hot(target, n_class).float()

        pred_error = pred_teacher.detach().clone() - pred_student
        #positives = pred_teacher > 0
        #pred_error *= positives
        one_cold = 1 - one_hot
        pred_error *= one_cold
        # loss for student network
        compensation_loss = pred_error.pow(2).sum() / (target.shape[0] * n_class - 1)

        prob_student = pred_student.softmax(dim=1)
        prob_teacher = pred_teacher.softmax(dim=1)
        prob_agree = prob_teacher * prob_student
        prob_agree = prob_agree / prob_agree.sum(dim=1, keepdims=True)
        prob_teacher_container[index], prob_student_container[index], prob_agree_container[index] = \
            prob_teacher[torch.arange(target.numel()).to(target.device), target].detach(),\
            prob_student[torch.arange(target.numel()).to(target.device), target].detach(), \
            prob_agree[torch.arange(target.numel()).to(target.device), target].detach()

        max_student_prediction = prob_student.max(dim=1)[0].detach()
        mean_max_student_prediction += max_student_prediction.sum()
        number_of_samples += max_student_prediction.numel()

        if group_1 is not None:
            # Robust training
            balance = torch.ones_like(target).float()
            balance[group_1[index]] = 0.7
            balance[group_2[index]] = 0.55
            balance[group_3[index]] = 0.45
            balance[group_4[index]] = 0.3
            robust_target = (1 - balance[:, None]) * one_hot + balance[:, None] * prob_student.detach().clone()
            # Force the network to minimize Entropy
            robust_target = robust_target.pow(1 + balance[:, None])
            target = robust_target / robust_target.sum(dim=1, keepdims=True)
        else:
            # Standard CE training
            target = one_hot

        naive_loss = F.cross_entropy(pred_teacher, target, reduction="none").mean()

        # compensation_loss.backward()
        # optimizer_student.step()
        # optimizer_student.zero_grad()

        naive_loss.backward()
        optimizer_teacher.step()
        optimizer_teacher.zero_grad()

        print(f"\rEpoch: {current_epoch:3.0f}/{args.n_epoch - 1:3.0f}\tIter: {batch_idx:3.0f}/{len(train_dataset) // batch_size - 1:3.0f}\tnaive_loss: {naive_loss:8.5f}\tcompensation_l2_loss: {compensation_loss:8.5f}", end='', flush=False)
    print()
    # New values not necessary -> return None values
    if group_1 is not None:
        return None, None, None, None
    # Return statistics for overfitting epoch calculation
    mean_max_student_prediction = mean_max_student_prediction / number_of_samples
    return mean_max_student_prediction, prob_teacher_container, prob_student_container, prob_agree_container

def split_dataset(p):
    # Convert logits to probabilities
    up_mean, up_sigma, lo_mean, lo_sigma, thresh = otsu(p)
    # Search if we are in overfitting region
    group_1 = p < lo_mean
    group_2 = ~group_1 * (p < thresh)
    group_4 = p >= up_mean
    group_3 = ~group_4 * (p >= thresh)
    return group_1, group_2, group_3, group_4


# Evaluate the Model
# Test the Model on clean test_data -- acc not used for algorithm
def my_evaluate(loader, teacher, student):
    teacher.eval()
    student.eval()

    correct = 0
    total = 0
    for images, labels, _ in loader:
        images = Variable(images).to(args.device)
        pred_teacher = teacher(images)
        pred_student = student(images)
        p_teacher = pred_teacher.softmax(dim=1)
        p_student = pred_student.softmax(dim=1)
        p_agree = p_teacher * p_student
        p_agree = p_agree / p_agree.sum(dim=1, keepdims=True)
        #p_agree = p_teacher
        p_id_agree = torch.argmax(p_agree, dim=1)

        total += labels.size(0)
        correct += (p_id_agree.cpu() == labels).sum()
    acc = 100 * float(correct) / float(total)

    return acc


def detect_overfitting_epoch(x):
    """ Detects the maximal average max likelihood of the student. This should be the start of overfitting """
    if len(x) < 6:
        return False
    for i in range(2, len(x) - 2):
        if x[i] > x[i - 1] and x[i] > x[i - 2] and x[i] > x[i + 1] and x[i] > x[i + 2]:
            return i
    return False


def otsu(p):
    """ Calculates a threshold to separate into two groups based on OTSU """
    hist = torch.histc(p, 1000, 0, 1)

    def Q(hist, s, mu):
        n1, n2 = hist[:s].sum(), hist[s:].sum()
        if n1 == 0 or n2 == 0:
            return -10000000, None, None, None, None
        mu1 = (hist[0:s] * (torch.arange(0, s).to(hist.device) / 1000)).sum() / n1
        mu2 = (hist[s:1000] * (torch.arange(s, 1000).to(hist.device) / 1000)).sum() / n2
        sigma1 = ((hist[0:s] * ((torch.arange(0, s).to(hist.device) / 1000) - mu1).pow(2)).sum() / n1).clamp(
            0.00001).sqrt()
        sigma2 = ((hist[s:1000] * ((torch.arange(s, 1000).to(hist.device) / 1000) - mu2).pow(2)).sum() / n2).clamp(
            0.00001).sqrt()
        q = (n1 * (mu1 - mu).pow(2) + n2 * (mu2 - mu).pow(2)) / (n1 * sigma1 + n2 * sigma2)
        return q, mu1, mu2, sigma1, sigma2

    q = [0, 0]
    for s in range(2, 998):
        q.append(Q(hist, s, p.mean())[0])
    s = torch.argmax(torch.tensor(q))
    q, mu1, mu2, sigma1, sigma2 = Q(hist, s, p.mean())
    mu2, sigma2, mu1, sigma1, s = mu2.detach().cpu().item(), sigma2.detach().cpu().item(), \
                                  mu1.detach().cpu().item(), sigma1.detach().cpu().item(), s / 1000
    return mu2, sigma2, mu1, sigma1, s


def save_models(teacher, student):
    # save state of teacher
    state_teacher = {'state_dict': teacher.state_dict(),
                     'epoch': epoch
                     }
    save_path_teacher = os.path.join('./', args.dataset + '_' + args.noise_type + '_teacher.pth.tar')
    torch.save(state_teacher, save_path_teacher)

    # save state of student
    state_student = {'state_dict': student.state_dict(),
                     'epoch': epoch
                     }
    save_path_student = os.path.join('./', args.dataset + '_' + args.noise_type + '_student.pth.tar')
    torch.save(state_student, save_path_student)


##################################### main code ################################################

# Seed
set_global_seeds(args.seed)
args.device = set_device()
print(args.device)
time_start = time.time()
# Hyper Parameters
batch_size = args.batch_size
learning_rate = args.lr
noise_type_map = {'clean': 'clean_label', 'worst': 'worse_label', 'aggre': 'aggre_label', 'rand1': 'random_label1',
                  'rand2': 'random_label2', 'rand3': 'random_label3', 'clean100': 'clean_label',
                  'noisy100': 'noisy_label'}
args.noise_type = noise_type_map[args.noise_type]
# load dataset
if args.noise_path is None:
    if args.dataset == 'cifar10':
        args.noise_path = './data/CIFAR-10_human.pt'
    elif args.dataset == 'cifar100':
        args.noise_path = './data/CIFAR-100_human.pt'
    else:
        raise NameError(f'Undefined dataset {args.dataset}')

if args.dataset=='cifar10':
    warm_up = 10
elif args.dataset=='cifar100':
    warm_up = 30

train_dataset, val_dataset, test_dataset, n_class, num_training_samples = input_dataset(args.dataset,
                                                                             args.noise_type,
                                                                             args.noise_path,
                                                                             is_human=True,
                                                                             val_ratio=args.val_ratio)
print('building model...')
teacher = PreResNet18(n_class)
teacher.to(args.device)
student = PreResNet18(n_class)
student.to(args.device)

print('building model done')
optimizer_teacher = torch.optim.SGD(list(teacher.parameters())+ list(student.parameters()), lr=learning_rate, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
optimizer_student = torch.optim.SGD(student.parameters(), lr=learning_rate, momentum=args.momentum,
                                    weight_decay=args.weight_decay)


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           num_workers=args.num_workers,
                                           shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=batch_size,
                                         num_workers=args.num_workers,
                                         shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          num_workers=args.num_workers,
                                          shuffle=False)


# Prediction container
prob_teacher_hist = list()
prob_student_hist = list()
prob_agreement_hist = list()
# Average max likelihood
avg_max_pred_student_hist = list()
# Epoch in which overfitting starts
overfitting_epoch = None
group_1, group_2, group_3, group_4 = None, None, None, None

test_acc = 0.0
best_acc = 0.0
# training
for epoch in range(args.n_epoch):
    # train models
    print(f'epoch {epoch}')
    # set learning rate according to shedule
    lr = args.lr
    if epoch >= 150:
        lr /= 10
    for param_group in optimizer_teacher.param_groups:
        param_group['lr'] = lr
    #for param_group in optimizer_student.param_groups:
    #    param_group['lr'] = lr

    # Train epoch and collect values for latter overfitting epoch calculation
    avg_max_pred_student, prob_teacher, prob_student, prob_agreement = my_train(
        epoch, teacher, optimizer_teacher, train_loader, student, optimizer_student, group_1, group_2, group_3, group_4)
    avg_max_pred_student_hist.append(avg_max_pred_student)
    prob_teacher_hist.append(prob_teacher)
    prob_student_hist.append(prob_student)
    prob_agreement_hist.append(prob_agreement)
    # Check if overfitting epoch needs to be predicted
    if not overfitting_epoch:
        overfitting_epoch = detect_overfitting_epoch(avg_max_pred_student_hist)
        if overfitting_epoch:
            group_1, group_2, group_3, group_4 = split_dataset(prob_agreement_hist[overfitting_epoch])
            np.save('detection.npy', (group_1 == 1).cpu().numpy())

    if epoch % args.clean_test_freq == 0 or epoch == args.n_epoch - 1:
        test_acc = my_evaluate(test_loader, teacher, student)
        if test_acc > best_acc:
            best_acc = test_acc
        print(f'epoch: {epoch}\t acc on clean test_data: {test_acc}\tbest: {best_acc}')

    if epoch % args.save_model_freq == 0 or epoch == args.n_epoch - 1:
        save_models(teacher, student)

    time_curr = time.time()
    time_elapsed = time_curr - time_start
    print(
        f'[Epoch {epoch}] Time elapsed {time_elapsed // 3600:.0f}h {(time_elapsed % 3600) // 60:.0f}m {(time_elapsed % 3600) % 60:.0f}s',
        flush=True)


