"""Main module."""

#############################################################
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import time
import pandas as pd
from tqdm.auto import tqdm
#############################################################

def get_validation_accuracy(args, model, val_loader, desc=None, leave=True):
    if desc is None:
        desc = f"Evaluating {args.model_name}"

    model = model.to(args.device)
    n_val_stop = args.n_val_stop
    if n_val_stop==0: n_val_stop = len(val_loader.dataset)

    model.eval()
    with torch.no_grad():

        correct_predictions = 0
        total_predictions = 0
        # running_loss_val = 0.0

        outer_progress = tqdm(val_loader, desc=desc, total=n_val_stop//args.batch_size, leave=leave)

        for images, true_labels in outer_progress:
            images = images.to(args.device)
            true_labels = true_labels.to(args.device)

            # Get predictions (no need for gradients)
            outputs = model(images)
            _, predicted_labels = torch.max(outputs, dim=1)

            # loss = criterion(outputs, true_labels)
            # running_loss_val += loss.item() * images.size(0)

            # Check if the prediction was correct for the entire batch
            # The comparison produces a tensor of booleans (True/False)
            correct_predictions_in_batch = (predicted_labels == true_labels)

            # Sum the boolean tensor to get the number of correct predictions in the batch
            # .item() extracts the number from the tensor
            correct_predictions += correct_predictions_in_batch.sum().item()

            # The total number of predictions is the batch size
            total_predictions += true_labels.size(0)
            # if total_predictions > n_val_stop: break # early stopping


        acc_val = correct_predictions / total_predictions
        # loss_val = running_loss_val / total_predictions
        outer_progress.set_postfix_str(f"accuracy={acc_val:.3f}")

    return acc_val #, loss_val

class StochasticDepth(nn.Module):
    def __init__(self, survival_prob=1.0):
        super().__init__()
        self.survival_prob = survival_prob

    def forward(self, x, residual):
        if self.training:
            if torch.rand(1).item() < self.survival_prob:
                return x + residual
            else:
                return x
        else:
            return x + self.survival_prob * residual

def train_model(args, model, train_loader, val_loader, df_train=None, 
                model_filename=None, json_filename=None):
    
    model = model.to(args.device)
    # retraining the full model
    for param in model.parameters():
        param.requires_grad = True        

    # sets the optimizer
    if args.delta2 > 0.: 
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(1-args.delta1, 1-args.delta2), 
                                    weight_decay=args.weight_decay) 
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=1-args.delta1, 
                                    weight_decay=args.weight_decay) # to set training variables
 
    n_train_stop = args.n_train_stop
    if n_train_stop==0: n_train_stop = len(train_loader.dataset)
    n_val_stop = args.n_val_stop
    if n_val_stop==0: n_val_stop = len(val_loader.dataset)


    # # Learning rate scheduler (cosine decay with warmup)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer,
    #     T_max=300 - 20,  # Total epochs minus warmup
    #     eta_min=1e-5
    # )

    # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html 
    # criterion = nn.CrossEntropyLoss(label_smoothing=
    )
    # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html 
    criterion = nn.BCEWithLogitsLoss()
    num_classes = train_loader.dataset.num_classes


    # the DataFrame to record from
    if df_train is None:
        i_epoch_start = 0
        if args.verbose: print(f"Starting learning...")
    else:
        i_epoch_start = df_train['epoch'].max() + 1
        if args.verbose: print(f"Starting from epoch {i_epoch_start} with {len(df_train)} records")
        # # reset the index
        # df_train.reset_index(drop=True, inplace=True)
        # make a copy of the DataFrame to avoid modifying the original one
        df_train = df_train.copy()

    since = time.time()
    # history = []
    max_acc_train, max_acc_val = 0., 0.
    total_image = 0
    outer_progress = tqdm(range(i_epoch_start, args.num_epochs), desc="Epochs", leave=True, disable=(args.num_epochs==1))
    for i_epoch in outer_progress:
        running_loss = 0.0
        running_corrects = 0
        i_image = 0
        # training on one set
        inner_progress = tqdm(train_loader, desc=f'Epoch={i_epoch+1}/{args.num_epochs}', 
                              total=n_train_stop//args.batch_size, leave=False)
        for images, true_labels in inner_progress:

            model.train()
            images, true_labels = images.to(args.device), true_labels.to(args.device)
            total_image += len(images)
            i_image += len(images)
            # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#use-parameter-grad-none-instead-of-model-zero-grad-or-optimizer-zero-grad
            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            _, predicted_labels = torch.max(outputs, dim=1)
            running_corrects += (predicted_labels == true_labels).sum().item()
             
            true_labels_onehot = nnf.one_hot(true_labels, num_classes=num_classes).float()
            true_labels_onehot = args.label_smoothing + (1-args.label_smoothing)*true_labels_onehot
            loss = criterion(outputs, true_labels_onehot)
            running_loss += loss.item() * images.size(0)
            loss.backward()
            optimizer.step()


        loss_train = running_loss / i_image
        acc_train = running_corrects*1. / i_image

        # validation on the ohter set
        acc_val = get_validation_accuracy(args, model, val_loader, leave=False)
        max_acc_train, max_acc_val = max((max_acc_train, acc_train)), max((max_acc_val, acc_val))

        outer_progress.set_postfix_str(f"Acc: train={acc_train:.3f} - val={acc_val:.3f} - (Max:train={max_acc_train:.3f} - val={max_acc_val:.3f})")
        
        result = [{'epoch': i_epoch, 'i_image':i_image, 'total_image':total_image, 'loss_train':loss_train, 'acc_train':acc_train, 'acc_val':acc_val, 'time':time.time() - since}] # 'loss_val':loss_val, 
        
        # save everything at each epoch
        if not(model_filename is None):
            # if args.verbose:  print(f"Saving...{model_filename}")
            torch.save(model.state_dict(), model_filename)

        if df_train is None:
            df_train = pd.DataFrame(result)
        else:
            df_new_row = pd.DataFrame(result)
            df_train = pd.concat([df_train, df_new_row], ignore_index=True)
        if not(json_filename is None):
            df_train.to_json(json_filename, orient='records', indent=2)

    return model, df_train

def do_learning(args, dataset, name, model_filename_init=None):


    model_filename = args.data_cache / f'{name}.pth'
    json_filename = args.data_cache / model_filename.name.replace('.pth', '.json')
    lock_filename = args.data_cache / model_filename.name.replace('.pth', '.lock')

    # %rm {lock_filename}  # FORCING RECOMPUTE

    df_train = None
    should_resume_training = not lock_filename.exists()

    if json_filename.exists():
        print(f"Load JSON from pre-trained resnet {json_filename}")
        df_train = pd.read_json(json_filename, orient='records')
        print(f"{model_filename}: accuracy = {df_train['acc_val'][-5:].mean():.3f}")
        should_resume_training = (df_train['epoch'].max() + 1 < args.num_epochs) and (not lock_filename.exists())

    if should_resume_training:
        lock_filename.touch() # as we do a training let's lock it
        from .torch_utils import get_loader, get_dataset, load_model

        TRAIN_DATA_DIR = args.DATAROOT / f'Imagenet_{dataset}' / 'train'
        train_dataset = get_dataset(args, TRAIN_DATA_DIR, n_stop=args.n_train_stop)
        train_loader = get_loader(args, train_dataset)
        VAL_DATA_DIR = args.DATAROOT / f'Imagenet_{dataset}' / 'val'
        val_dataset = get_dataset(args, VAL_DATA_DIR, n_stop=args.n_val_stop)
        val_loader = get_loader(args, val_dataset)

        # we need to train the model or finish a training that already started
        print(f"Training model {args.model_name}, file= {model_filename} - image_size={args.image_size}")

        if model_filename.is_file(): 
            model_filename_train = model_filename
        else:
            model_filename_train = model_filename_init # we use a stored file for learning or None

        model = load_model(args, model_filename=model_filename_train)

                
        start_time = time.time()
        model_retrain, df_train = train_model(args, model=model, train_loader=train_loader, val_loader=val_loader, df_train=df_train, model_filename=model_filename, json_filename=json_filename)
        elapsed_time = time.time() - start_time
        print(f"Training completed in {elapsed_time // 60:.0f}m {elapsed_time % 60:.0f}s")

    if lock_filename.exists(): lock_filename.unlink()
    return model_filename, json_filename

# from torchvision.transforms.functional import crop, resize
from .torch_utils import get_preprocess
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

def compute_likelihood_map(args, model, full_image,
                            resolution = (15, 15), # how many fixation points to use
                            size_ratio = 0.618 # how much of the image to use relative to radius                        
                            ):

    _, H, W = full_image.shape
    # max_size = np.max((H, W))
    min_size = np.min((H, W))
    aspect_ratio = H/W
    box_size = int(min_size*size_ratio)
    N_fixations = np.prod(resolution)
    resolution = (int(np.sqrt(N_fixations*aspect_ratio)), int(np.sqrt(N_fixations/aspect_ratio)))
    N_fixations = np.prod(resolution)

    pos_h = np.linspace(0, H, resolution[0]+2, endpoint=True)[1:-1]
    pos_w = np.linspace(0, W, resolution[1]+2, endpoint=True)[1:-1]
    pos_H, pos_W = np.meshgrid(pos_h, pos_w)

    # args.image_size = box_size
    preprocess = get_preprocess(args)
    pil_image = TF.to_pil_image(full_image)

    cropped_images = torch.empty((N_fixations, 3, args.image_size, args.image_size))
    for i_fixation, (h, w) in enumerate(zip(pos_H.ravel(), pos_W.ravel())):
        h, w = int(h), int(w)
        cropped = TF.crop(pil_image, h-box_size//2, w-box_size//2, box_size, box_size)
        # resized = TF.resize(cropped, [args.image_size, args.image_size], interpolation=InterpolationMode.BILINEAR, antialias=True)
        cropped_images[i_fixation, ...] = preprocess(cropped)
 
    with torch.no_grad():
        cropped_images = cropped_images.to(args.device)
        outputs = nnf.softmax(model(cropped_images), dim=1)

    return pos_H, pos_W, outputs
