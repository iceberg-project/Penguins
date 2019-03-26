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
    parser.add_argument('--num_cycles', type=int, nargs='?', help='number of training cycles')
    parser.add_argument('--num_epochs', type=int, nargs='?', help='number of epochs per training cycle')
    parser.add_argument('--cycle_mult', type=int, nargs='?', help='multiplier for cycle length from the '
                                                                  'second cycle onwards')
    parser.add_argument('--loss_func', type=str, default='MSE')
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


def train_model(model, dataloader, criterion, optimizer, scheduler, num_epochs, loss_name,
                model_name, models_dir, binary_target, learning_rate=1E-3, num_cycles=3, cycle_mult=2):
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
    :param learning_rate:
    :param num_cycles:
    :return:
    """
    # set model name and path
    model_name = f"{model_name}_binary-{binary_target}_loss-{loss_name}_lr-{learning_rate}_ep-{num_epochs}_nc-{num_cycles}"
    model_path = f"{models_dir}/{model_name}"
    os.makedirs(model_path, exist_ok=True)

    # create summary writer with tensorboardX
    writer = SummaryWriter(log_dir='./tensorboard_logs/{}_{}'.format(model_name, str(datetime.datetime.now())))

    # keep track of iterations
    global_step = 0

    # keep track of best loss
    best_loss = 10E8

    # set cuda
    use_gpu = torch.cuda.is_available()

    # run training cycles
    for cycle in range(num_cycles):
        # reset learning rate
        optim_cycle = optimizer
        sched_cycle = scheduler
        sched_cycle.T_max = scheduler.T_max * max(1, (cycle_mult * cycle))

        # each cycle has n epochs
        for epoch in range(num_epochs * max(1, (cycle_mult * cycle))):
            epoch_loss = 0
            exp_avg_loss = 0
            # training and validation loops
            for phase in ["training", "validation"]:
                for iter, data in enumerate(dataloader[phase]):
                    if phase == "training":
                        # add global step
                        global_step += 1

                        # zero gradients
                        optim_cycle.zero_grad()

                        # step with scheduler
                        scheduler.step()

                        # get input data
                        input_img, target_img, area = data

                        # change area to binary
                        if binary_target:
                            area = torch.Tensor([cnt > 0 for cnt in area])
                            area.requires_grad = True

                        if use_gpu:
                            input_img, target_img, area = input_img.cuda(), target_img.cuda(), area.cuda()

                        # get model predictions
                        preds = model(input_img)

                        # get loss
                        if "Area" in model_name:
                            loss = criterion(preds, area)
                        else:
                            loss = criterion(preds.view(preds.numel()), target_img.view(target_img.numel()))

                        exp_avg_loss = 0.99 * exp_avg_loss + 0.1 * (loss.item() / len(preds))

                        # update parameters
                        loss.backward()
                        optim_cycle.step()

                        # save stats
                        if iter > 0 and iter % 500 == 0:
                            writer.add_scalar("training loss", exp_avg_loss, global_step)
                            writer.add_scalar("learning rate", optim_cycle.param_groups[-1]['lr'], global_step)

                    else:
                        with torch.no_grad():
                            # get input data
                            input_img, target_img, area = data

                            # set target to binary
                            if binary_target:
                                area = torch.Tensor([cnt > 0 for cnt in area])

                            # cuda
                            if use_gpu:
                                input_img, target_img, area = input_img.cuda(), target_img.cuda(), area.cuda()

                            # get model predictions
                            preds = model(input_img)

                            # get loss
                            if "Area" in model_name:
                                loss = criterion(preds, area)
                            else:
                                loss = criterion(preds.view(preds.numel()), target_img.view(target_img.numel()))

                            epoch_loss += loss.item() / len(preds)

            if phase == "validation":
                epoch_loss /= (len(dataloader["validation"]))
                writer.add_scalar("validation loss", epoch_loss, global_step)
                is_best_loss = epoch_loss < best_loss
                best_loss = min(epoch_loss, best_loss)
                save_checkpoint(model_name, model.state_dict(), is_best_loss)

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

    #TODO add weighted sampler
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
    criterion = loss_functions[args.loss_func]
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.1)
    sched = args.scheduler
    if sched == 'Cosine':
        scheduler = schedulers[sched](optimizer, dataloaders['training'])
    elif sched == 'Step':
        scheduler = schedulers[sched](optimizer, args.hyp_set)

    if use_gpu:
        model = model.cuda()
        model = nn.DataParallel(model)
        criterion = criterion.cuda()

    train_model(model=model, dataloader=dataloaders, criterion=criterion,
                optimizer=optimizer, scheduler=scheduler, num_cycles=args.num_cycles,
                num_epochs=args.num_epochs, model_name=model_name, loss_name=args.loss_func,
                models_dir=args.models_dir, binary_target=args.binary_target)


if __name__ == "__main__":
    main()
