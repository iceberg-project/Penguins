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


def lr_find(model, dataloader, optimizer):
    return None
    # TODO learning rate finder


def save_checkpoint(filename, state, is_best_loss):
    torch.save(state, filename + '.tar')
    if is_best_loss:
        shutil.copyfile(filename + '.tar', filename + '_best_loss.tar')


def train_model(model, dataloader, criterion_seg, criterion_reg, optimizer, scheduler, num_epochs, loss_name,
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
    model_name = f"{model_name}_ts-{ts_name}_binary-{binary_target}_loss-{loss_name}_lr-{learning_rate}_ep-{num_epochs}"
    model_path = f"{models_dir}/{model_name}"
    os.makedirs(model_path, exist_ok=True)
    print(f'\n Training {model_name}')

    # keep track of iterations
    global_step = 0

    # keep track of best dice loss
    dice_loss = 0
    n_masks = 0
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

        epoch_loss = 0
        epoch_dice = 0
        exp_avg_loss = 0
        # training and validation loops
        for phase in ["training", "validation"]:
            print('\n{} \n'.format(phase))
            for iter, data in enumerate(dataloader[phase]):
                if phase == "training":
                    # add global step
                    global_step += 1

                    # zero gradients
                    optimizer.zero_grad()

                    # step with scheduler
                    scheduler.step()

                    # get input data
                    input_img, target_img, area, is_mask = data

                    # transform area to tensor
                    if binary_target:
                        area = torch.Tensor([cnt > 0 for cnt in area])

                    else:
                        area = torch.Tensor(area)

                    is_mask = is_mask.reshape([len(is_mask), 1, 1, 1])

                    if use_gpu:
                        input_img, target_img, area, is_mask = input_img.cuda(), target_img.cuda(), area.cuda(), is_mask.cuda()

                    # get model predictions
                    pred_mask, pred_area = model(input_img)

                    # get loss
                    if "Area" in model_name:
                        loss_area = criterion_reg(pred_area, area)
                    else:
                        loss_area = False

                    # keep only true mask entries for segmentation loss
                    pred_mask = pred_mask * is_mask
                    target_img = target_img * is_mask

                    if pred_mask:
                        loss_seg = criterion_seg(pred_mask.view(pred_mask.numel()), target_img.view(target_img.numel())) * sum(is_mask) / len(pred_mask)
                    else:
                        loss_seg = False

                    # add up losses and update parameters
                    loss = 0
                    if loss_area:
                        loss = loss + loss_area
                    elif loss_seg:
                        loss = loss + loss_seg

                    if loss:
                        loss.backward()
                    else:
                        loss = torch.Tensor([0])

                    optimizer.step()

                    # store loss
                    exp_avg_loss = 0.99 * exp_avg_loss + 0.1 * loss.item()

                    # save stats
                    if iter > 0 and iter % 100 == 0:
                        writer.add_scalar("training loss", exp_avg_loss, global_step)
                        writer.add_scalar("learning rate", optimizer.param_groups[-1]['lr'], global_step)

                else:
                    with torch.no_grad():
                        # get input data
                        input_img, target_img, area, is_mask = data

                        # set target to binary
                        if binary_target:
                            area = torch.Tensor([cnt > 0 for cnt in area])

                        else:
                            area = torch.Tensor(area)

                        is_mask = is_mask.reshape([len(is_mask), 1, 1, 1])

                        # cuda
                        if use_gpu:
                            input_img, target_img, area = input_img.cuda(), target_img.cuda(), area.cuda()

                        # get model predictions
                        pred_mask, pred_area = model(input_img)

                        # get loss
                        if "Area" in model_name:
                            loss_area = criterion_reg(pred_area, area)
                        else:
                            loss_area = torch.Tensor([0])

                        # filter images to keep masks
                        pred_mask = pred_mask * is_mask
                        target_img = target_img * is_mask

                        if pred_mask:
                            loss_seg = criterion_seg(pred_mask.view(pred_mask.numel()),
                                                     target_img.view(target_img.numel())) * sum(is_mask) / len(pred_mask)
                        else:
                            loss_seg = torch.Tensor([0])

                        # get epoch loss and DICE for segmentation
                        loss = loss_area + loss_seg
                        epoch_loss += loss.item()
                        if pred_mask:
                            epoch_dice += dice_metric(pred_mask, target_img).item() * sum(is_mask) / len(pred_mask)
                            n_masks += sum(is_mask)

        if phase == "validation":
            if n_masks:
                epoch_dice /= n_masks
            else:
                epoch_dice = 9999
            epoch_loss /= (len(dataloader["validation"]))
            writer.add_scalar("validation loss", epoch_loss, global_step)
            writer.add_scalar("validation DICE", epoch_dice, global_step)
            is_best_loss = epoch_dice < best_loss
            best_loss = min(epoch_loss, best_loss)
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
    pos_weight, neg_weight = len(classes) / sum(classes), len(classes) / (len(classes) - sum(classes))
    weights = [pos_weight * ele + neg_weight * (1 - ele) for ele in classes]
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

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
                model_name=model_name, loss_name=args.loss_funcs,
                models_dir=args.models_dir, binary_target=args.binary_target)


if __name__ == "__main__":
    main()
