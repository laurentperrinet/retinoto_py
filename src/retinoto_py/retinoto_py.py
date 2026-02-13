"""Main module."""

#############################################################
from .torch_utils import get_loader, get_dataset, load_model

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.optim.lr_scheduler import LambdaLR

import time
import pandas as pd
from tqdm.auto import tqdm
# from timm.data.mixup import Mixup
# from timm.loss import SoftTargetCrossEntropy

# from torchvision.transforms.functional import crop, resize
from .torch_utils import get_preprocess, fixate
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
#############################################################

def get_validation_accuracy(args, model, val_loader, desc=None, leave=True):
    if desc is None:
        desc = f"Evaluating {args.model_name}"

    model = model.to(args.device)
    model.eval()
    with torch.no_grad():

        correct_predictions = 0
        total_predictions = 0
        outer_progress = tqdm(val_loader, desc=desc, total=len(val_loader.dataset)//args.batch_size, leave=leave)

        for images, true_idxs in outer_progress:
            images = images.to(args.device)
            true_idxs = true_idxs.to(args.device)

            # Get predictions (no need for gradients)
            outputs = model(images)
            _, predicted_true_idxs = torch.max(outputs, dim=1)

            # Check if the prediction was correct for the entire batch
            # The comparison produces a tensor of booleans (True/False)
            correct_predictions_in_batch = (predicted_true_idxs == true_idxs)

            # Sum the boolean tensor to get the number of correct predictions in the batch
            # .item() extracts the number from the tensor
            correct_predictions += correct_predictions_in_batch.sum().item()

            # The total number of predictions is the batch size
            total_predictions += true_idxs.size(0)

        acc_val = correct_predictions / total_predictions
        outer_progress.set_postfix_str(f"accuracy={acc_val:.3f}")

    return acc_val

def get_optimizer(args, model):
    optim_dict = dict(lr=args.base_lr, weight_decay=args.weight_decay)
    assert(0 <= args.delta1 <= 1)
    assert(0 <= args.delta2 <= 1)
    if args.optimizer_name=='adam': 
        optimizer = torch.optim.Adam(model.parameters(), betas=(1-args.delta1, 1-args.delta2), **optim_dict)
    elif args.optimizer_name=='adamw': 
        optimizer = torch.optim.AdamW(model.parameters(), betas=(1-args.delta1, 1-args.delta2), **optim_dict)
    elif args.optimizer_name=='sgd': 
        optimizer = torch.optim.SGD(model.parameters(),  momentum=1-args.delta1, dampening=1-args.delta2, **optim_dict)
    elif args.optimizer_name=='rmsprop': 
        optimizer = torch.optim.RMSprop(model.parameters(), momentum=1-args.delta1, alpha=1-args.delta2, **optim_dict)
    # elif args.optimizer_name=='adagrad': 
    #     optimizer = torch.optim.Adagrad(model.parameters(), betas=(1-args.delta1, 1-args.delta2), **optim_dict)
    elif args.optimizer_name=='adadelta': 
        optimizer = torch.optim.Adadelta(model.parameters(), rho=1-args.delta1, **optim_dict)
    else:
        raise(ValueError(f'Unknown optimizer {args.optimizer_name}'))

    return optimizer

class NegLogitLoss(nn.Module):
    """
    loss = - 1/B * Σ  output[batch_i, true_idx_i]

    *output*  : (B, C) raw logits from the model (no soft‑max)
    *true_idx*: (B,) integer class indices 0 … C‑1

    """
    def __init__(self, reduction="mean"):
        """
        reduction  – "mean" (default) returns the average over the batch,
                     "sum"  returns the sum,
                     "none" returns a vector of shape (B,).
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, outputs: torch.Tensor, true_idxs: torch.Tensor) -> torch.Tensor:
        # `true_idxs` is shape (B,); we need it as column vector for gather
        true_idxs = true_idxs.view(-1, 1)               # (B,1)

        # Gather the logits that belong to the true class:
        #   torch.gather returns a tensor of shape (B,1) with the selected values
        true_logits = torch.gather(outputs, dim=1, index=true_idxs)  # (B,1)

        # Negate them
        loss = -true_logits.squeeze(1)                  # (B,)

        # Apply the requested reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:   # "none"
            return loss

def get_cosine_schedule_with_warmup(optimizer, num_warmup_epochs, num_epochs, rel_final_lr):
    def lr_lambda(current_epoch):
        if current_epoch < num_warmup_epochs:
            # Constant warmup of 1
            return 1
        else:
            # Cosine decay from base_lr to final_lr
            progress = (current_epoch - num_warmup_epochs) / max(1, num_epochs - num_warmup_epochs)
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress)) # from 1 to zero
            return (cosine_decay + rel_final_lr) / (1 + rel_final_lr) # between 1 and down to rel_final_lr

    scheduler = LambdaLR(optimizer, lr_lambda, last_epoch=-1)
    return scheduler


def train_model(args, train_loader, val_loader, df_train=None, 
                model_filename=None, json_filename=None):

    # sets the model and optimizer
    model = load_model(args)
    model = model.to(args.device)
    optimizer = get_optimizer(args, model)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                args.num_warmup_epochs, args.num_epochs, args.final_lr/args.base_lr)

    # the DataFrame to record from
    if df_train is None:
        i_epoch_start = 0
        if args.verbose: print("Starting learning...")
    else:
        i_epoch_start = df_train['epoch'].max() + 1
        if args.verbose: print(f"Starting from epoch {i_epoch_start} with {len(df_train)} records")
        # checkpoint = torch.load(model_filename)
        checkpoint = torch.load(model_filename, map_location=torch.device(args.device), weights_only=False)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        df_train = df_train.copy()
        for _ in range(i_epoch_start):
            scheduler.step()

    if args.do_full_training:
        # retraining the full model
        for param in model.parameters():
            param.requires_grad = True

    else:
        # Freeze everything except FC layer
        for name, param in model.named_parameters():
            if not name.startswith('classifier'):
                param.requires_grad = False
            else:
                param.requires_grad = True

 
    # Using reduction='mean' to automatically scale the gradient for different batch sizes
    if args.loss_name=='CrossEntropyLoss':
        # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html 
        criterion = torch.nn.CrossEntropyLoss(reduction='mean', label_smoothing=args.label_smoothing)
    elif args.loss_name=='NegLogitLoss':
        # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html 
        # TODO add the logit of the chance level to normalize and form an odd ratio
        criterion = NegLogitLoss(reduction='mean')
    elif args.loss_name=='BCEWithLogitsLoss':
        # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html 
        criterion = nn.BCEWithLogitsLoss(reduction='mean')

    since = time.time()
    total_image = 0
    num_classes = len(train_loader.dataset.classes)
    outer_progress = tqdm(range(i_epoch_start, args.num_epochs), desc="Epochs",
                          leave=True, disable=((args.num_epochs-i_epoch_start)==1))
    for i_epoch in outer_progress:
        running_loss = 0.0
        running_corrects = 0
        i_image = 0
        inner_progress = tqdm(train_loader, desc=f'Epoch={i_epoch+1}/{args.num_epochs}',
                              total=len(train_loader.dataset)//args.batch_size, leave=False)
        model.train()
        for images, true_idxs in inner_progress:
            images, true_idxs = images.to(args.device), true_idxs.to(args.device)

            total_image += len(images)
            i_image += len(images)
            # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#use-parameter-grad-none-instead-of-model-zero-grad-or-optimizer-zero-grad
            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)

            if args.loss_name=='BCEWithLogitsLoss':
                true_idxs_onehot = nnf.one_hot(true_idxs, num_classes=num_classes).float()
                true_idxs_onehot = args.true_idx_smoothing/num_classes + (1-args.label_smoothing)*true_idxs_onehot
                loss = criterion(outputs, true_idxs_onehot)
            else:
                loss = criterion(outputs, true_idxs)
            loss.backward()
            optimizer.step()

            _, predicted_true_idxs = torch.max(outputs, dim=1)
            running_corrects += (predicted_true_idxs == true_idxs).sum().item()
            running_loss += loss.item() * images.size(0)

        scheduler.step()
        # print(f'DEBUG - lr={optimizer.param_groups[0]["lr"]:.2e} - epo')
        loss_train = running_loss / i_image
        acc_train = running_corrects*1. / i_image

        # validation on the ohter set
        acc_val = get_validation_accuracy(args, model, val_loader, leave=False)

        # save everything at each epoch
        if model_filename is not None:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, model_filename)

        result = [{'epoch': i_epoch, 'i_image':i_image, 'total_image':total_image, 
                   'loss_train':loss_train, 'acc_train':acc_train, 'acc_val':acc_val, 
                   'time':time.time() - since}] # 'loss_val':loss_val, 
        if df_train is None:
            df_train = pd.DataFrame(result)
        else:
            df_new_row = pd.DataFrame(result)
            df_train = pd.concat([df_train, df_new_row], ignore_index=True)
        if json_filename is not None:
            df_train.to_json(json_filename, orient='records', indent=2)

        postfix_str = f"Acc: train={acc_train:.3f} - val={acc_val:.3f} - "
        postfix_str += f"(Max:train={df_train['acc_train'].max():.3f} - val={df_train['acc_val'].max():.3f})"
        outer_progress.set_postfix_str(postfix_str)

    return model, df_train

def do_learning(args, dataset, name):

    model_filename = args.data_cache / f'{name}.pth'
    json_filename = args.data_cache / f'{name}.json'
    lock_filename = args.data_cache / f'{name}.lock'

    # %rm {lock_filename}  # FORCING RECOMPUTE

    df_train = None
    should_resume_training = not lock_filename.exists() # sets first this to True if there is no lock file

    if json_filename.exists():
        print(f"Load JSON from pre-trained resnet {json_filename}")
        df_train = pd.read_json(json_filename, orient='records')
        print(f"{model_filename}: latest accuracy = {df_train.tail(1)['acc_val'].item():.3f}")
        # resume learning if we still have some epochs to run
        should_resume_training = (df_train['epoch'].max() + 1 < args.num_epochs)

    if should_resume_training:
        lock_filename.touch() # as we do a training, let's lock it

        TRAIN_DATA_DIR = args.DATAROOT / f'Imagenet_{dataset}' / 'train'
        train_dataset = get_dataset(args, TRAIN_DATA_DIR, do_augment=args.do_augment)
        train_loader = get_loader(args, train_dataset)
        VAL_DATA_DIR = args.DATAROOT / f'Imagenet_{dataset}' / 'val'
        val_dataset = get_dataset(args, VAL_DATA_DIR, do_augment=False)
        val_loader = get_loader(args, val_dataset)

        # we need to train the model or finish a training that already started
        print(f"Training model {args.model_name}, file= {model_filename} - image_size={args.image_size}")

        start_time = time.time()
        model_retrain, df_train = train_model(args,
                                              train_loader, val_loader, df_train, model_filename, json_filename)
        elapsed_time = time.time() - start_time
        print(f"Training of {model_retrain} completed in {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s")

    if lock_filename.exists(): lock_filename.unlink()
    return model_filename, json_filename


def get_positions(H, W, resolution=(15, 15), endpoint=False, do_hex=True):

    if endpoint:
        pos_h = np.linspace(0, H, resolution[0], endpoint=True)
        pos_w = np.linspace(0, W, resolution[1], endpoint=True)
    else:
        pos_h = np.linspace(0, H, resolution[0]+2, endpoint=True)[1:-1]
        pos_w = np.linspace(0, W, resolution[1]+2, endpoint=True)[1:-1]

    pos_H, pos_W = np.meshgrid(pos_h, pos_w)

    if do_hex:
        if H<W:
            delta = (pos_h[1]-pos_h[0])/4
            pos_H[::2] += delta
            pos_H[1::2] -= delta
        else:
            delta = (pos_w[1]-pos_w[0])/4
            pos_W[::2] += delta
            pos_W[1::2] -= delta

    pos_H, pos_W = pos_H.ravel(), pos_W.ravel()
    return pos_H, pos_W

def compute_likelihood_map(args, model, full_image,
                           pos_H, pos_W,
                           size_ratio = 0.618, # how much of the image to use relative to radius
                           do_min_boxsize = False
                           ):

    three, H, W = full_image.shape
    assert three == 3
    if do_min_boxsize:
        # max_size = np.max((H, W))
        min_size = np.min((H, W))
        box_size = int(min_size*size_ratio)
    else:
        box_size = int(np.sqrt(H*W)*size_ratio)

    # take a smaller box if the image is small
    # box_size = min(())
    # args.image_size = box_size
    preprocess = get_preprocess(args, do_augment=False)
    # preprocess = preprocess.to(args.device)
    # pil_image = TF.to_pil_image(full_image)

    N_fixations = len(pos_H)
    assert N_fixations == len(pos_W)

    gaze_images = torch.empty((N_fixations, 3, args.image_size, args.image_size))
    for i_fixation, (h, w) in enumerate(zip(pos_H, pos_W)):
        h, w = int(h), int(w) 
        image_fix = fixate(full_image, h, w, box_size).to(args.device)
        gaze_images[i_fixation, ...] = preprocess(image_fix)

    with torch.no_grad():
        gaze_images = gaze_images.to(args.device)
        # probas = nnf.sigmoid(model(gaze_images))
        probas = nnf.softmax(model(gaze_images), dim=1)

    return probas
