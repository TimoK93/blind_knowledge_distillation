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
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.02)
parser.add_argument('--val_ratio', type=float, default=0.0)
parser.add_argument('--noise_type', type=str, help='clean, aggre, worst, rand1, rand2, rand3, clean100, noisy100',
                    default='aggre')
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
args = parser.parse_args()


# # Train the Model
# def train(epoch, train_loader, model, optimizer):
#     train_total = 0
#     train_correct = 0
#     model.train()
#     for i, (images, labels, indexes) in enumerate(train_loader):
#
#         batch_size = indexes.shape[0]
#
#         images = images.to(args.device)
#         labels = labels.to(args.device)
#
#         # Forward + Backward + Optimize
#         logits = model(images)
#
#         prec, _ = accuracy(logits, labels, topk=(1, 5))
#         train_total += 1
#         train_correct += prec
#         loss = F.cross_entropy(logits, labels, reduce=True)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         if (i + 1) % args.print_freq == 0:
#             print('Epoch [%d/%d], Iter [%d/%d] Training Accuracy: %.4F, Loss: %.4f'
#                   % (epoch + 1, args.n_epoch, i + 1, len(train_dataset) // batch_size, prec, loss.data))
#
#     train_acc = float(train_correct) / float(train_total)
#     return train_acc


def my_train(current_epoch, teacher, optimizer_teacher, dataloader, student, optimizer_student):
    # teacher.train()
    # student.train()
    num_iter = (len(dataloader.dataset)//dataloader.batch_size)+1
    for batch_idx, (img, target, index) in enumerate(dataloader):
        img = img.to(args.device)
        target = target.to(args.device)
        loss = dict(loss=0)

        # Load modified labels
        if current_epoch == 0:
            print(f'img: {img.device}')
            print(f'corrected_target: {corrected_target.device}')
            one_hot = F.one_hot(target, n_class).float()
            corrected_target[index] = target.detach()
            corrected_target_one_hot[index] = one_hot.detach()
            original_target[index] = target.detach()
        else:
            target = corrected_target[index]

        img, target = img.to(args.device), target.to(args.device)
        _pred_teacher = teacher(img)
        _pred_student = student(img)

        one_hot = F.one_hot(target, n_class).float()
        _is_true_label = is_true_label[index]
        _use_for_training = use_for_training[index]
        if _use_for_training.sum() == 0:
            _use_for_training[0:round(_use_for_training.shape[0] / 2)] = 1

        # pred_error = pred_teacher.detach().clone() - pred_student
        pred_error = _pred_teacher.detach().clone() - _pred_student
        positives = _pred_teacher > 0
        pred_error *= positives
        one_cold = 1 - one_hot
        pred_error *= one_cold
        pred_error *= _use_for_training[:, None]
        # loss for student network
        compensation_loss = pred_error.pow(2).sum() / (_use_for_training.sum() * (n_class - 1))
        loss["compensation_l2"] = compensation_loss
        loss["loss"] += compensation_loss

        alpha = current_epoch / args.n_epoch
        alpha = min(alpha + 0.2, args.alpha_final)
        if not overfitting_epoch:
            alpha = 0.0
        prob_student = _pred_student.softmax(dim=1)
        balance = alpha * torch.ones_like(_is_true_label)
        if overfitting_epoch:
            # Bootstrap target
            if is_true_label.sum() > 0:
                # balance = 1 - self.is_true_label[index]
                if args.trusted_to_0:
                    balance = torch.where(_is_true_label == 1, torch.zeros_like(balance), balance)
                if args.untrusted_to_1 and not args.incremental_label_correction:
                    balance = torch.where(_is_true_label == 0, torch.ones_like(balance), balance)
            if args.incremental_label_correction:
                _one_hot = one_hot
                _one_hot[_is_true_label == 0] = corrected_target_one_hot[index].clone()[is_true_label == 0]
            else:
                _one_hot = one_hot
            bootstrapped_target = (1 - balance[:, None]) * _one_hot + balance[:, None] * prob_student
            # Force the network to minimize Entropy
            bootstrapped_target = bootstrapped_target.pow(1 + alpha)
            _target = bootstrapped_target / bootstrapped_target.sum(dim=1, keepdims=True)
        else:
            _target = one_hot
        if args.incremental_label_correction:
            corrected_target_one_hot[index] = 0.95 * corrected_target_one_hot[index].detach() + 0.05 * _target.detach()
        _pred_teacher, __target = _pred_teacher[_use_for_training == 1], _target[_use_for_training == 1]
        naive_loss = F.cross_entropy(_pred_teacher, __target, reduction="none",
                                     label_smoothing=args.label_smoothing).mean()
        loss["ce"] = naive_loss
        loss["loss"] += naive_loss

        compensation_loss.backward()
        optimizer_student.step()
        optimizer_student.zero_grad()

        naive_loss.backward()
        optimizer_teacher.step()
        optimizer_teacher.zero_grad()


        # sys.stdout.write('%s:%.1f-%s | Epoch [%3d/%3d] Iter[%3d/%3d]\t CE-loss: %.4f \t l2-loss: %.4f'
        #                  % (args.dataset, args.r, args.cifar_n_mode, epoch, args.num_epochs, batch_idx + 1, num_iter,
        #                     naive_loss.item(), compensation_loss.item()))
        # sys.stdout.flush()

        #sys.stdout.write('\r')
        sys.stdout.write(
            f"Epoch: {current_epoch:3.0f}/{args.n_epoch - 1:3.0f}\tIter: {batch_idx:3.0f}/{len(train_dataset) // batch_size - 1:3.0f}\tLoss: {loss['loss']:8.5f}\tnaive_loss: {loss['ce']:8.5f}\tcompensation_l2_loss: {loss['compensation_l2']:8.5f}\n")
        sys.stdout.flush()

    return loss


def on_epoch_end(current_epoch, is_true_label, otsu_thresh, otsu_mu2, p_overfitting, overfitting_epoch):
    if not args.training:
        return is_true_label, otsu_thresh, otsu_mu2, p_overfitting, overfitting_epoch
    # Convert logits to probabilities
    p_teacher = pred_teacher.softmax(dim=1)
    p_id_teacher = torch.argmax(p_teacher, dim=1)
    p_student = pred_student.softmax(dim=1)
    p_agree = p_teacher * p_student
    p_agree = p_agree / p_agree.sum(dim=1, keepdims=True)
    p_gt_agree = p_agree[torch.arange(corrected_target.numel()), corrected_target]
    p_id_student = torch.argmax(p_student, dim=1)
    p_id_agree = torch.argmax(p_agree, dim=1)
    # Calculate average max prediction of student
    max_p_comp = p_student.max(dim=1)[0].mean().detach()
    avg_max_pred_student.append(max_p_comp)

    # Search if we are in overfitting region
    if not overfitting_epoch:
        overfitting_epoch = detect_overfitting_epoch(avg_max_pred_student)
        if overfitting_epoch:
            # Calculate Otsu
            _p_teacher = pred_teacher_hist[overfitting_epoch].softmax(dim=1)
            _p_student = pred_student_hist[overfitting_epoch].softmax(dim=1)
            _p_agree = _p_teacher * _p_student
            _p_agree = _p_agree / _p_agree.sum(dim=1, keepdims=True)
            _p_gt_agree = _p_agree[torch.arange(corrected_target.numel()), corrected_target]
            up_mean, up_sigma, lo_mean, lo_sigma, thresh = otsu(_p_gt_agree)
            otsu_thresh, otsu_mu2 = thresh, up_mean
            p_overfitting = _p_gt_agree.clone()
            if args.stop_after_overfitting:
                trainer_should_stop = True
            print("up_mean, up_sigma, lo_mean, lo_sigma, thresh", up_mean, up_sigma, lo_mean, lo_sigma, thresh)

    # Perform label correction
    if not args.divide_dataset or not overfitting_epoch:
        return is_true_label, otsu_thresh, otsu_mu2, p_overfitting, overfitting_epoch
    # Split dataset in save and unsafe data for the first time
    if is_true_label.sum() == 0:
        if args.split_dataset == 'otsu_thresh':
            is_true_label = (p_gt_agree >= otsu_thresh) * 1
        elif args.split_dataset == 'otsu_up_mean':
            is_true_label = (p_gt_agree >= otsu_mu2) * 1
        else:
            raise Exception('no valid argument for split_dataset')
        if args.incremental_label_correction and args.incremental_label_correction_label_reset:
            corrected_target_one_hot[is_true_label] = 1 / n_class
    if not args.correct_labels or not overfitting_epoch:
        return is_true_label, otsu_thresh, otsu_mu2, p_overfitting, overfitting_epoch
    # Correct data for unsafe labels and detect new save objects
    if (current_epoch % 5) == 0 and current_epoch > 0 and otsu_thresh is not None:
        # Find labels that should be modified
        is_true_label[is_true_label == 0] = (p_gt_agree[is_true_label == 0] >= otsu_mu2) * 1
    return is_true_label, otsu_thresh, otsu_mu2, p_overfitting, overfitting_epoch


# # Evaluate the Model
# def evaluate(loader, model, save=False, best_acc=0.0):
#     model.eval()  # Change model to 'eval' mode.
#
#     correct = 0
#     total = 0
#     for images, labels, _ in loader:
#         images = Variable(images).to(args.device)
#         logits = model(images)
#         outputs = F.softmax(logits, dim=1)
#         _, pred = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (pred.cpu() == labels).sum()
#     acc = 100 * float(correct) / float(total)
#     if save:
#         if acc > best_acc:
#             state = {'state_dict': model.state_dict(),
#                      'epoch': epoch,
#                      'acc': acc,
#                      }
#             save_path = os.path.join('./', args.dataset + '_' + args.noise_type + 'best.pth.tar')
#             torch.save(state, save_path)
#             best_acc = acc
#             print(f'model saved to {save_path}!')
#
#     return acc

# Evaluate the Model
def my_evaluate(loader, model, save=False, best_acc=0.0):
    model.eval()  # Change model to 'eval' mode.

    correct = 0
    total = 0
    for images, labels, _ in loader:
        images = Variable(images).to(args.device)
        logits = model(images)
        outputs = F.softmax(logits, dim=1)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred.cpu() == labels).sum()
    acc = 100 * float(correct) / float(total)
    if save:
        if acc > best_acc:
            state = {'state_dict': model.state_dict(),
                     'epoch': epoch,
                     'acc': acc,
                     }
            save_path = os.path.join('./', args.dataset + '_' + args.noise_type + 'best.pth.tar')
            torch.save(state, save_path)
            best_acc = acc
            print(f'model saved to {save_path}!')

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
# print('train_labels:', len(train_dataset.train_labels), train_dataset.train_labels[:10])
# load model
print('building model...')
teacher = PreResNet18(n_class)
teacher.to(args.device)
student = PreResNet18(n_class)
student.to(args.device)

print('building model done')
optimizer_teacher = torch.optim.SGD(teacher.parameters(), lr=learning_rate, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
optimizer_student = torch.optim.SGD(student.parameters(), lr=learning_rate, momentum=args.momentum,
                                    weight_decay=args.weight_decay)

# CE = nn.CrossEntropyLoss(reduction='none')
# CE_loss = nn.CrossEntropyLoss()

# all_loss = [[], []]  # save the history of losses from two networks

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



alpha_plan = [0.1] * 50 + [0.01] * 50
# Annotations container
original_target = -torch.ones(50000).long().to(args.device)
corrected_target = -torch.ones(50000).long().to(args.device)
corrected_target_one_hot = -torch.ones(50000, n_class).float().to(args.device)
corrected_target_hist = torch.zeros((args.n_epoch, 50000)).long().to(args.device)
# Prediction container
pred_teacher = torch.zeros((50000, n_class)).float().to(args.device)
pred_student = torch.zeros((50000, n_class)).float().to(args.device)
pred_teacher_hist = torch.zeros((args.n_epoch, 50000, n_class)).float().to(args.device)
pred_student_hist = torch.zeros((args.n_epoch, 50000, n_class)).float().to(args.device)
is_true_label = torch.zeros(50000).float().to(args.device)
# Average max likelihood
avg_max_pred_student = list()
# Epoch in which overfitting starts
overfitting_epoch = False
# Otsu threshold
otsu_thresh, otsu_mu2 = None, None
# Probabilities at the overfitting epoch
p_overfitting = None
# flag to perform early stopping
stop_after_overfitting = False
# flag to randomly deactivate samples in the training
use_for_training = torch.ones(50000).long().to(args.device)


train_acc = 0.0
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
    for param_group in optimizer_student.param_groups:
        param_group['lr'] = lr

    # if not is_overfitted:
    #     # warum-up training pre overfitting-epoch without bootstrapping
    #     warmup_trainloader = loader.run('warmup')
    #     print('warm up')
    #     warmup(epoch, teacher, optimizer_teacher, warmup_trainloader, student, optimizer_student)
    # else:
    #     # training post overfitting-epoch with bootstrapping
    #     pass

    my_train(epoch, teacher, optimizer_teacher, train_loader, student, optimizer_student)
    is_true_label, otsu_thresh, otsu_mu2, p_overfitting, overfitting_epoch = on_epoch_end(epoch, is_true_label, otsu_thresh, otsu_mu2, p_overfitting, overfitting_epoch)

    # if overfitting_epoch and otsu_thresh is None:
    #     mu2, sigma2, mu1, sigma1, thresh = otsu(p_agree)


    # # train models
    # train_acc = train(epoch, train_loader, model, optimizer)
    # print('train acc is ', train_acc)
    # # evaluate models
    # print('previous_best', best_acc)
    # if args.val_ratio > 0.0:
    #     # save results
    #     val_acc = evaluate(loader=val_loader, model=model, save=True, best_acc=best_acc)
    #     if val_acc > best_acc:
    #         best_acc = val_acc
    #     print('validation acc is ', val_acc)
    # test_acc = evaluate(loader=test_loader, model=model, save=False, best_acc=best_acc)
    # print('test acc is ', test_acc)
    time_curr = time.time()
    time_elapsed = time_curr - time_start
    print(
        f'[Epoch {epoch}] Time elapsed {time_elapsed // 3600:.0f}h {(time_elapsed % 3600) // 60:.0f}m {(time_elapsed % 3600) % 60:.0f}s',
        flush=True)
