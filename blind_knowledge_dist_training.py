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
                    default='worst')
parser.add_argument('--noise_path', type=str, help='path of CIFAR-10_human.pt', default=None)
parser.add_argument('--dataset', type=str, help=' cifar10 or cifar100', default='cifar10')
parser.add_argument('--n_epoch', type=int, default=300)
parser.add_argument('--seed', type=int,
                    default=0)  # we will test your code with 5 different seeds. The seeds are generated randomly and fixed for all participants.
parser.add_argument('--print_freq', type=int, default=100)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--batch_size', default=128, type=int, help='train batch-size')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD optimizer')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay for SGD optimizer')
parser.add_argument('--clean_test_freq', type=int, default=10, help='how often the current model performance on clean data will be displayed')
parser.add_argument('--noisy_val_freq', type=int, default=10, help='how often the current model performance on noisy data will be displayed')

parser.add_argument('--alpha1', type=float, default=0.30, help='alpha for group1 (highest confidence in gt labels)')
parser.add_argument('--alpha2', type=float, default=0.45, help='alpha for group2 (higher confidence in gt labels)')
parser.add_argument('--alpha3', type=float, default=0.55, help='alpha for group3 (lower confidence in gt labels)')
parser.add_argument('--alpha4', type=float, default=0.70, help='alpha for group4 (lowest confidence in gt labels)')
args = parser.parse_args()


def train(current_epoch, teacher, optimizer, dataloader, student, group_1, group_2, group_3, group_4):
    prob_teacher_container, prob_student_container, prob_agree_container = \
        torch.zeros((len(dataloader.dataset))).to(args.device), \
        torch.zeros((len(dataloader.dataset))).to(args.device),\
        torch.zeros((len(dataloader.dataset))).to(args.device)
    mean_max_student_prediction = 0
    number_of_samples = 0
    last_iter_printed = 0
    batch_idx = 0
    for batch_idx, (img, target, index) in enumerate(dataloader):
        img = img.to(args.device)
        target = target.to(args.device)

        # Load modified labels
        img, target = img.to(args.device), target.to(args.device)
        pred_teacher = teacher(img)
        pred_student = student(img)

        one_hot = F.one_hot(target, n_class).float()

        pred_error = pred_teacher - pred_student
        one_cold = 1 - one_hot
        pred_error *= one_cold
        # loss for student network
        compensation_loss = pred_error.pow(2).sum() / (target.shape[0] * n_class - 1)

        prob_student = pred_student.softmax(dim=1)
        prob_teacher = pred_teacher.softmax(dim=1)
        prob_agree = prob_teacher * prob_student
        prob_agree = prob_agree / prob_agree.sum(dim=1, keepdims=True)
        prob_teacher_container[index], prob_student_container[index], prob_agree_container[index] = \
            prob_teacher[torch.arange(target.numel()).to(target.device), target].detach().clone(),\
            prob_student[torch.arange(target.numel()).to(target.device), target].detach().clone(), \
            prob_agree[torch.arange(target.numel()).to(target.device), target].detach().clone()

        max_student_prediction = prob_student.max(dim=1)[0].detach()
        mean_max_student_prediction += max_student_prediction.sum()
        number_of_samples += max_student_prediction.numel()

        if group_1 is not None:
            # Robust training
            balance = torch.ones_like(target).float()
            balance[group_1[index]] = args.alpha1
            balance[group_2[index]] = args.alpha2
            balance[group_3[index]] = args.alpha3
            balance[group_4[index]] = args.alpha4
            robust_target = (1 - balance[:, None]) * one_hot + balance[:, None] * prob_student
            # Force the network to minimize Entropy
            robust_target = robust_target.pow(1 + balance[:, None])
            target = robust_target / robust_target.sum(dim=1, keepdims=True)
        else:
            # Standard CE training
            target = one_hot

        naive_loss = F.cross_entropy(pred_teacher, target, reduction="none").mean()

        loss = naive_loss + compensation_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.print_freq == 0:
            last_iter_printed = batch_idx
            print(f"epoch: {current_epoch:3.0f}/{args.n_epoch - 1:3.0f}\titer: {batch_idx:3.0f}/{len(train_dataset) // batch_size:3.0f}\tloss: {loss:8.5f}\tnaive_loss: {naive_loss:8.5f}\tcompensation_l2_loss: {compensation_loss:8.5f}")
    if last_iter_printed != batch_idx:
        print(f"epoch: {current_epoch:3.0f}/{args.n_epoch - 1:3.0f}\titer: {batch_idx:3.0f}/{len(train_dataset) // batch_size:3.0f}\tloss: {loss:8.5f}\tnaive_loss: {naive_loss:8.5f}\tcompensation_l2_loss: {compensation_loss:8.5f}")
    print()
    # New values not necessary -> return None values
    if group_1 is not None:
        return None, None, None, None
    # Return statistics for overfitting epoch calculation
    mean_max_student_prediction = mean_max_student_prediction / number_of_samples
    return mean_max_student_prediction, prob_teacher_container, prob_student_container, prob_agree_container


def calc_noise_probability(teacher, dataloader, student):
    prob_agree_container = list()
    teacher.eval()
    student.eval()
    with torch.no_grad():
        for batch_idx, (img, target, index) in enumerate(dataloader):
            img = img.to(args.device)
            target = target.to(args.device)
            pred_teacher = teacher(img)
            pred_student = student(img)

            prob_student = pred_student.softmax(dim=1)
            prob_teacher = pred_teacher.softmax(dim=1)
            prob_agree = prob_teacher * prob_student
            prob_agree = prob_agree / prob_agree.sum(dim=1, keepdims=True)
            prob_agree = prob_agree[torch.arange(target.numel()).to(target.device), target]
            prob_agree_container.append(prob_agree.detach().clone())

    prob_agree_container = torch.cat(prob_agree_container)

    return prob_agree_container


def split_dataset(p):
    # Convert logits to probabilities
    up_mean, up_sigma, lo_mean, lo_sigma, thresh = otsu(p)
    print(f'otsu split performed:\n\tup_mean: {up_mean:0.5f}\tup_sigma: {up_sigma:0.5f}\n\tlo_mean: {lo_mean:0.5f}\tlo_sigma: {lo_sigma:0.5f}\n\tthresh:  {thresh:0.5f}\n')
    # Search if we are in overfitting region
    group_4 = p < lo_mean
    group_3 = ~group_4 * (p < thresh)
    group_1 = p >= up_mean
    group_2 = ~group_1 * (p >= thresh)

    return group_1, group_2, group_3, group_4


# Evaluate the Model
# Test the Model on clean test_data -- acc not used for algorithm
def evaluate(loader, teacher, student):
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
        p_id_agree = torch.argmax(p_agree, dim=1)

        total += labels.size(0)
        correct += (p_id_agree.cpu() == labels).sum()
    acc = 100 * float(correct) / float(total)

    return acc


def detect_overfitting_epoch(x):
    """ Detects the maximal average max likelihood of the student. This should be the start of overfitting """

    if len(x) < 6:
        return False
    if len(x) == 10:
        return 8
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
    save_path_teacher = os.path.join(save_path, 'teacher.pth.tar')
    torch.save(state_teacher, save_path_teacher)

    # save state of student
    state_student = {'state_dict': student.state_dict(),
                     'epoch': epoch
                     }
    save_path_student = os.path.join(save_path, 'student.pth.tar')
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
save_path = os.path.join('./results/', args.dataset + '_' + args.noise_type + '_seed_' + str(args.seed))
os.makedirs(save_path, exist_ok=True)
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

train_dataset_full, train_dataset, val_dataset, test_dataset, n_class, num_training_samples = input_dataset(args.dataset,
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

optimizer = torch.optim.SGD(list(teacher.parameters()) + list(student.parameters()), lr=learning_rate, momentum=args.momentum,
                            weight_decay=args.weight_decay)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.1)


train_loader_full = torch.utils.data.DataLoader(dataset=train_dataset_full,  # Is used to create detection.npy. Not used for training or validation!
                                           batch_size=batch_size,
                                           num_workers=args.num_workers,
                                           shuffle=False)

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
# Results for noise detection task
prob_agreement_hist_DETECTION_TASK = list()
# Average max likelihood
avg_max_pred_student_hist = list()
# Epoch in which overfitting starts
overfitting_epoch = None
group_1, group_2, group_3, group_4 = None, None, None, None

test_acc = 0.0
val_acc = 0.0
best_test_acc = [0.0, 0]
best_val_acc = [0.0, 0]
# training
for epoch in range(args.n_epoch):
    # train models
    print(f'epoch {epoch}')

    teacher.train()
    student.train()
    # Train epoch and collect values for latter overfitting epoch calculation
    avg_max_pred_student, prob_teacher, prob_student, prob_agreement = train(
        epoch, teacher, optimizer, train_loader, student, group_1, group_2, group_3, group_4)
    teacher.eval()
    student.eval()

    avg_max_pred_student_hist.append(avg_max_pred_student)
    prob_teacher_hist.append(prob_teacher)
    prob_student_hist.append(prob_student)
    prob_agreement_hist.append(prob_agreement)
    # Check if overfitting epoch needs to be predicted
    if not overfitting_epoch:
        overfitting_epoch = detect_overfitting_epoch(avg_max_pred_student_hist)
        if overfitting_epoch:
            print()
            print('##################################################')
            print(f'overfitting epoch: {overfitting_epoch}')
            print('##################################################')
            print()
            # Split dataset for next step: Robust training
            group_1, group_2, group_3, group_4 = split_dataset(prob_agreement_hist[overfitting_epoch])
            # DETECTION TASK: Save un-shuffled full dataset and save detection.npy. After this, the full dataset will not be used again.
            group_1_full, group_2_full, group_3_full, group_4_full = split_dataset(prob_agreement_hist_DETECTION_TASK[overfitting_epoch])
            save_path_detection = os.path.join(save_path, 'detection.npy')
            np.save(save_path_detection, group_4_full.cpu().numpy())
        else:
            full_dataset_noise_probability = calc_noise_probability(teacher, train_loader_full, student)
            prob_agreement_hist_DETECTION_TASK.append(full_dataset_noise_probability)

    # evaluate + save models if new best val_acc
    if epoch % args.noisy_val_freq == 0 or epoch == args.n_epoch - 1:
        if args.val_ratio == 0.0:
            val_acc = evaluate(train_loader, teacher, student)
        else:
            val_acc = evaluate(val_loader, teacher, student)
        if val_acc > best_val_acc[0]:
            print(f'models saved - new best val_acc!  (prev best: {best_val_acc[0]:0.2f} on epoch: {best_val_acc[1]})')
            best_val_acc[0] = val_acc
            best_val_acc[1] = epoch
            save_models(teacher, student)
        print(f'epoch: {epoch}\tacc on noisy val_data:  {val_acc:0.2f}\tbest: {best_val_acc[0]:0.2f} (epoch: {best_val_acc[1]})')

    # test
    if epoch % args.clean_test_freq == 0 or epoch == args.n_epoch - 1:
        test_acc = evaluate(test_loader, teacher, student)
        if test_acc > best_test_acc[0]:
            best_test_acc[0] = test_acc
            best_test_acc[1] = epoch
        print(f'epoch: {epoch}\tacc on clean test_data: {test_acc:0.2f}\tbest: {best_test_acc[0]:0.2f} (epoch: {best_test_acc[1]})')

    # save models on training end
    if epoch == args.n_epoch - 1:
        save_models(teacher, student)
        print('models saved - end of training')

    time_curr = time.time()
    time_elapsed = time_curr - time_start
    print(
        f'epoch: {epoch}\ttime elapsed: {time_elapsed // 3600:.0f}h {(time_elapsed % 3600) // 60:.0f}m {(time_elapsed % 3600) % 60:.0f}s',
        flush=True)
    print()

    scheduler.step()
