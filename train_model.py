import argparse
import datetime
import os
import time
import shutil

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
from PIL import ImageFile

from utils.dataloaders.dataloader_train import ImageFolderTrain
from utils.dataloaders.transforms import TransformPair
from utils.model_library import *
from utils.loss_functions import DiceLoss

ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args():
    parser = argparse.ArgumentParser(description='trains a CNN to find seals in satellite imagery')
    parser.add_argument('--t_dir', type=str, help='base directory to recursively search for images in')
    parser.add_argument('--model_arch', type=str, help='model architecture, must be a member of models '
                                                       'dictionary')
    parser.add_argument('--hyp_set', type=str, help='combination of hyperparameters used, must be a member of '
                                                    'hyperparameters dictionary')
    parser.add_argument('--models_dir', type=str, default='saved_models', help='folder where the model will be saved')
    parser.add_argument('--lr', type=float, nargs='?', help='learning rate for training')
    parser.add_argument('--num_epochs', type=int, nargs='?', help='number of epochs per training cycle')
    parser.add_argument('--loss_funcs', type=str, default='BCE-MSE')
    parser.add_argument('--binary_target', type=int, default=0)
    parser.add_argument('--scheduler', type=str, default='Cosine')
    return parser.parse_args()


def save_checkpoint(filename, state, is_best_loss):
    torch.save(state, filename + '.tar')
    if is_best_loss:
        shutil.copyfile(filename + '.tar', filename + '_best_loss.tar')


def get_iou(pred, target, thresh):
    pred = (pred > thresh)
    target = target.to(torch.uint8)

    intersection = torch.sum(pred & target).item()
    union = torch.sum(pred | target).item()

    return (intersection + 1E-6) / (union + 1E-6)


def train_model(model, dataloader, criterion_seg, criterion_reg, optimizer, scheduler, sched_name, num_epochs,
                loss_name,
                model_name, models_dir, binary_target, ts_name, learning_rate=1E-3):
    """

    :param model:
    :param data:
    :param criterion:
    :param optimizer:
    :param scheduler:
    :param num_epochs:
    :param loss_name:
    :param model_name:
    :param models_dir:
    # create summary writer with tensorboardX
    writer = SummaryWriter(log_dir='./tensorboard_logs/{}_{}'.format(model_name, str(datetime.datetime.now())))
    :param learning_rate:
    :param num_cycles:
    :return:
    """
    # set model name and path
    model_name = f"{model_name}_ts-{ts_name}_binary-{binary_target}_loss-{loss_name}_lr-{learning_rate}_schd-{sched_name}_ep-{num_epochs}"
    model_path = f"{models_dir}/{model_name}"
    os.makedirs(model_path, exist_ok=True)
    print(f'\n Training {model_name}')

    # keep track of iterations
    global_step = 0

    # keep track of best dice loss
    best_loss = 10E8

    # validation metric -- DICE loss
    dice_metric = DiceLoss()

    # set cuda
    use_gpu = torch.cuda.is_available()

    # setup tensorboard
    writer = SummaryWriter(log_dir='./tensorboard_logs/{}_{}'.format(model_name, str(datetime.datetime.now())))

    # each cycle has n epochs
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        if sched_name == "Step":
            scheduler.step()

        epoch_loss = 0
        epoch_dice = 0
        exp_avg_loss_area = 0
        exp_avg_loss_mask = 0

        # store intersection and union for IoU 50, 75 and 90
        iou_treshs = [0.25, 0.5, 0.75, 0.9]
        epoch_iou = [0] * len(iou_treshs)

        # training and validation loops
        for phase in ["training", "validation"]:
            print('\n{} \n'.format(phase))
            for iter, data in enumerate(dataloader[phase]):
                if "training" in phase:
                    # add global step
                    global_step += 1

                    # zero gradients
                    optimizer.zero_grad()

                    # step with scheduler
                    if sched_name == 'Cosine':
                        scheduler.step()

                    if phase == "training" and 'Area' in model_name:
                        # get input data for area
                        input_img, target_img, area, label = data

                        # only keep target images with the correct class for segmentation
                        idcs = [idx for idx, ele in enumerate(label) if ele != 1]
                        target_img = target_img[idcs, :, :, :]
                        # transform area to tensor
                        if binary_target:
                            area = torch.Tensor([cnt > 0 for cnt in area])

                        else:
                            area = torch.Tensor(area)

                        if use_gpu:
                            input_img, target_img, area = input_img.cuda(), target_img.cuda(), area.cuda()

                        # get model predictions
                        pred_mask, pred_area = model(input_img)

                        # filter true masks
                        pred_mask = pred_mask[idcs, :, :, :]

                        # get loss for regression and segmentation
                        loss = criterion_reg(pred_area, area)
                        loss_seg = criterion_seg(pred_mask, target_img)

                        # save stats
                        if iter > 0 and iter % 10 == 0:
                            writer.add_scalar(f"training loss area {loss_name.split('-')[-1]}", loss,
                                              global_step)
                            writer.add_scalar(f"training loss mask {loss_name.split('-')[0]}", loss_seg,
                                              global_step)
                            writer.add_scalar("learning rate", optimizer.param_groups[-1]['lr'], global_step)

                        # store loss
                        exp_avg_loss_area = 0.99 * exp_avg_loss_area + 0.01 * loss.item()
                        exp_avg_loss_mask = 0.99 * exp_avg_loss_mask + 0.01 * loss_seg.item()

                        # backprop
                        loss = loss + loss_seg
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()


                    elif phase == "training" and 'Area' not in model_name:
                        # get inputs for segmentation
                        input_img, target_img, _, label = data

                        # only keep target images with the correct class for segmentation
                        idcs = [idx for idx, ele in enumerate(label) if ele != 1]
                        target_img = target_img[idcs, :, :, :]

                        if use_gpu:
                            input_img, target_img = input_img.cuda(), target_img.cuda()

                        # get model predicitions
                        pred_mask, _ = model(input_img)

                        # filter true masks
                        pred_mask = pred_mask[idcs, :, :, :]

                        # loss
                        loss = criterion_seg(pred_mask.view(pred_mask.numel()),
                                             target_img.view(target_img.numel()))

                        # store loss
                        exp_avg_loss_mask = 0.99 * exp_avg_loss_mask + 0.01 * loss.item()

                        if iter > 0 and iter % 10 == 0:
                            writer.add_scalar(f"training loss mask {loss_name.split('-')[0]}", loss,
                                              global_step)
                            writer.add_scalar("learning rate", optimizer.param_groups[-1]['lr'], global_step)

                        # backprop
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()


                else:
                    with torch.no_grad():
                        # store iou
                        # get input data -- we only care about segmentation here, so all images are true masks
                        input_img, target_img, area, _ = data

                        # set target to binary
                        if binary_target:
                            area = torch.Tensor([cnt > 0 for cnt in area])

                        else:
                            area = torch.Tensor(area)

                        # cuda
                        if use_gpu:
                            input_img, target_img, area = input_img.cuda(), target_img.cuda(), area.cuda()

                        # get model predictions
                        pred_mask, pred_area = model(input_img)

                        # get loss
                        if "Area" in model_name:
                            loss = criterion_reg(pred_area, area)
                        else:
                            loss = torch.Tensor([0])

                        loss_seg = criterion_seg(pred_mask.view(pred_mask.numel()),
                                                 target_img.view(target_img.numel()))

                        # get epoch loss and DICE for segmentation
                        loss = loss + loss_seg
                        epoch_loss += loss.item()
                        epoch_dice += dice_metric(pred_mask, target_img).item()

                        # get iou at 50, 75 and 90
                        for idx, ele in enumerate(iou_treshs):
                            epoch_iou += get_iou(torch.sigmoid(pred_mask), target_img, ele)


        if phase == "validation":
            epoch_dice /= len(dataloader["validation"])
            epoch_loss /= len(dataloader["validation"])
            epoch_iou /= len(dataloader["validation"])
            writer.add_scalar("validation loss", epoch_loss, global_step)
            writer.add_scalar("validation DICE", epoch_dice, global_step)
            for idx, ele in enumerate(iou_treshs):
                writer.add_scalar(f"validation IoU-{ele}", epoch_iou, global_step)
            is_best_loss = epoch_dice < best_loss
            best_loss = min(epoch_dice, best_loss)
            save_checkpoint(model_path, model.state_dict(), is_best_loss)

    return model


def main():
    # unroll arguments
    args = parse_args()
    hyp_set = args.hyp_set

    # set cuda
    use_gpu = torch.cuda.is_available()

    # augmentation
    patch_size = model_archs[args.model_arch]
    data_transforms = {
        'training': TransformPair(patch_size, train=True),
        'validation': TransformPair(patch_size, train=False)
    }

    # load images
    image_datasets = {x: ImageFolderTrain(root=os.path.join(args.t_dir, x),
                                          patch_size=patch_size,
                                          transform=data_transforms[x])
                      for x in ['training', 'validation']}

    # weighted sampler
    classes = image_datasets['training'].classes

    def make_weights_for_balanced_classes(labels, nclasses):
        """
        Generates weights to get balanced classes during training. To be used with weighted random samplers.

        :param images: list of training images in training set.
        :param nclasses: number of classes on training set.
        :return: list of weights for each training image.
        """
        count = [0] * nclasses
        for lbl in labels:
            count[lbl] += 1
        weight_per_class = [0.] * nclasses
        N = float(sum(count))
        for i in range(nclasses):
            weight_per_class[i] = N / float(count[i])
        weight = [0] * len(labels)
        for idx, val in enumerate(labels):
            weight[idx] = weight_per_class[val]
        return weight

    weights = make_weights_for_balanced_classes(classes, len(set(classes)))
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, 3000)

    dataloaders = {"training": torch.utils.data.DataLoader(image_datasets["training"],
                                                           batch_size=
                                                           hyperparameters[hyp_set]['batch_size_train'],
                                                           num_workers=
                                                           hyperparameters[hyp_set][
                                                               'num_workers_train'],
                                                           sampler=sampler),
                   "validation": torch.utils.data.DataLoader(image_datasets["validation"],
                                                             batch_size=
                                                             hyperparameters[hyp_set]['batch_size_val'],
                                                             num_workers=
                                                             hyperparameters[hyp_set][
                                                                 'num_workers_val'])
                   }

    model = model_defs[args.model_arch]
    model_name = args.model_arch
    loss_funcs = args.loss_funcs.split('-')
    criterion_seg = loss_functions[loss_funcs[0]]
    criterion_reg = loss_functions[loss_funcs[1]]
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.1)
    sched = args.scheduler
    if sched == 'Cosine':
        scheduler = schedulers[sched](optimizer, dataloaders['training'])
    else:
        scheduler = schedulers[sched](optimizer, args.hyp_set)

    if use_gpu:
        model = model.cuda()
        model = nn.DataParallel(model)
        criterion_seg = criterion_seg.cuda()
        criterion_reg = criterion_reg.cuda()

    train_model(model=model, dataloader=dataloaders, criterion_seg=criterion_seg,
                criterion_reg=criterion_reg, ts_name=args.t_dir.split('_')[-1],
                optimizer=optimizer, scheduler=scheduler, num_epochs=args.num_epochs,
                sched_name=sched, model_name=model_name, loss_name=args.loss_funcs,
                models_dir=args.models_dir, binary_target=args.binary_target)


if __name__ == "__main__":
    main()
