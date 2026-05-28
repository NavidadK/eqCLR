import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
from escnn import gspaces

import numpy as np
import time
import pickle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from medmnist import PathMNIST, DermaMNIST, TissueMNIST, BloodMNIST
from torchvision.datasets import CIFAR10

from eqCLR.eq_resnet import EqResNet18, EqResNet, eq_resnet
from evaluation import model_eval, eval_knn_single, dataset_to_X_y, lin_eval_rep, should_eval

###################### PARAMS ##############################

folder = 'results/cifar10'
name_data = 'cifar10'
N_CLASSES = 10
IN_CHANNELS = 3
IMG_RESIZE = 33 

BATCH_SIZE = 512
N_EPOCHS = 100
N_CPU_WORKERS = 16
BASE_LR = 0.06         
WEIGHT_DECAY = 5e-4    
MOMENTUM = 0.9
PROJECTOR_HIDDEN_SIZE = 1024
PROJECTOR_OUTPUT_SIZE = 128
CROP_LOW_SCALE = 0.2
GRAYSCALE_PROB = 0.1   
PRINT_EVERY_EPOCHS = 1
EVAL_DURING_TRAIN = True
ITER_SAVE_EMBED = None
MAXPOOL = False 

for seed in range(3):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # data
    if IMG_RESIZE is not None:
        transform = transforms.Compose([
            transforms.Resize((IMG_RESIZE, IMG_RESIZE)),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.ToTensor()
    
    data_train = CIFAR10(root="data/cifar10", train=True, download=False, transform=transform)
    data_test = CIFAR10(root="data/cifar10", train=False, download=False, transform=transform)
    data_val = CIFAR10(root="data/cifar10", train=False, download=False, transform=transform)
    print("Data loaded.")

    if IMG_RESIZE is None:
        IMG_RESIZE = data_train[0][0].shape[1]  # get image size from dataset
    print(f"Image size (resized): {IMG_RESIZE}")

    transforms_ssl = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=IMG_RESIZE, scale=(CROP_LOW_SCALE, 1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
            ),
            transforms.RandomGrayscale(p=GRAYSCALE_PROB),
            transforms.ToTensor(), 
        ]
    )

    class PairedTransform:
        def __init__(self, transform):
            self.transform = transform

        def __call__(self, x):
            return (self.transform(x), self.transform(x))


    paired_ssl_transforms = PairedTransform(transforms_ssl)

    data_train_ssl = CIFAR10(root="data/cifar10", train=True, download=False, transform=paired_ssl_transforms)

    data_loader_ssl = DataLoader(
        data_train_ssl,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=N_CPU_WORKERS,
        pin_memory=True,
    )

    # eval data
    class RandomRightAngleRotation:
        """Randomly rotate PIL image by 90, 180, or 270 degrees."""
        def __call__(self, x):
            angle = int(torch.randint(1, 4, ()).item()) * 90
            return x.rotate(angle)
        
    transforms_classifier = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=IMG_RESIZE, scale=(CROP_LOW_SCALE, 1)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )

    data_train_classifier = CIFAR10(root="data/cifar10", train=True, download=False, transform=transforms_classifier)

    data_loader_classifier = DataLoader(
            data_train_classifier,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=N_CPU_WORKERS,
        )
    
    def infoNCE(features, temperature=0.5):
            x = F.normalize(features)
            cos_xx = x @ x.T / temperature
            cos_xx.fill_diagonal_(float("-inf"))
            
            batch_size = cos_xx.size(0) // 2
            targets = torch.arange(batch_size * 2, dtype=int, device=cos_xx.device)
            targets[:batch_size] += batch_size
            targets[batch_size:] -= batch_size

            return F.cross_entropy(cos_xx, targets)

    # EqCLR
    groups = ['flip_rot', 'rot']
    for group in groups:
        for N in [2, 4, 8]:

            ###################### NETWORK ARCHITECTURE #########################

            MODEL_FILENAME = f"C-0003_seed{seed}_{name_data}-eqCLR_resnet18_N{N}_no_maxpool_{group}_kp_{N_EPOCHS}epochs_lr{BASE_LR}"
            print(f"Model filename: {MODEL_FILENAME}")

            if group == 'flip_rot':
                model = EqResNet18(N=int(N/2), r2_act=gspaces.flipRot2dOnR2(int(N/2)), in_channels=IN_CHANNELS, maxpool=MAXPOOL, projector_hidden_size=PROJECTOR_HIDDEN_SIZE, n_classes=PROJECTOR_OUTPUT_SIZE, adjust_channels='keep_param')
            elif group == 'rot':
                model = EqResNet18(N=N, r2_act=gspaces.rot2dOnR2(N), in_channels=IN_CHANNELS, maxpool=MAXPOOL, projector_hidden_size=PROJECTOR_HIDDEN_SIZE, n_classes=PROJECTOR_OUTPUT_SIZE, adjust_channels='keep_param')

            optimizer = SGD(
                model.parameters(),
                lr=BASE_LR * BATCH_SIZE / 256,
                momentum=MOMENTUM,
                weight_decay=WEIGHT_DECAY,
            )

            scheduler = CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
            ###################### TRAINING LOOP #########################

            print("Starting training.")

            device = "cuda"

            model.to(device)
            model.train()
            knn_dict = {}
            embed_dict = {}
            training_start_time = time.time()
            time_train = 0.0

            for epoch in range(N_EPOCHS):
                epoch_loss = 0.0
                start_time = time.time()

                for batch_idx, batch in enumerate(data_loader_ssl):
                    views, _ = batch

                    views = [view.to(device, non_blocking=True) for view in views]

                    optimizer.zero_grad()

                    _, z1 = model(views[0])
                    _, z2 = model(views[1])
                    loss = infoNCE(torch.cat((z1, z2)))
                    epoch_loss += loss.item()

                    loss.backward()
                    optimizer.step()

                end_time = time.time()
                time_train += end_time - start_time

                scheduler.step()

                if EVAL_DURING_TRAIN and should_eval(epoch, n_epochs=N_EPOCHS):
                    model.eval()
                    with torch.no_grad():
                        X_train, y_train, Z_train = dataset_to_X_y(data_train, model)
                        X_test, y_test, Z_test = dataset_to_X_y(data_val, model)

                        knn_acc = eval_knn_single(X_train, y_train, X_test, y_test)
                        knn_dict[epoch] = knn_acc

                        if ITER_SAVE_EMBED is not None and epoch % ITER_SAVE_EMBED == 0:
                            embed_dict[epoch] = {
                                "X_test": X_test,
                                "y_test": y_test,
                            }
                    model.train()

                if epoch % PRINT_EVERY_EPOCHS == 0:
                    print(
                        f"Epoch {epoch + 1}, "
                        f"average loss {epoch_loss / len(data_loader_ssl):.4f}, "
                        f"{end_time - start_time:.1f} s",
                        f"KNN accuracy {knn_dict.get(epoch, 'N/A')}",
                        flush=True
                    )

            training_end_time = time.time()
            hours = (training_end_time - training_start_time) / 60 // 60
            minutes = (training_end_time - training_start_time) / 60 % 60
            average = (training_end_time - training_start_time) / N_EPOCHS
            print(
                f"Total training length for {N_EPOCHS} epochs: {hours:.0f}h {minutes:.0f}min",
                f"({average:.1f} sec/epoch)",
                flush=True
            )

            model.eval()
            torch.save(model.state_dict(), f'{folder}/model_weights/{MODEL_FILENAME}_weights.pt')
            print(f"Model saved to {MODEL_FILENAME}_weights.pt")

            model_details = {
                "Filename": MODEL_FILENAME,
                "Model structure": str(model),
                "N_EPOCHS": N_EPOCHS,
                "BATCH_SIZE": BATCH_SIZE,
                "BASE_LR": BASE_LR,
                "WEIGHT_DECAY": WEIGHT_DECAY,
                "MOMENTUM": MOMENTUM,
                "CROP_LOW_SCALE": CROP_LOW_SCALE,
                "GRAYSCALE_PROB": GRAYSCALE_PROB,
                "PROJECTOR_HIDDEN_SIZE": PROJECTOR_HIDDEN_SIZE,
                "PROJECTOR_OUTPUT_SIZE": PROJECTOR_OUTPUT_SIZE,
                "Training augmentations": str(transforms_ssl),
                "Training time": training_end_time - training_start_time,
                "Training time w/o evaluation": time_train,
                "MAXPOOL": MAXPOOL,
                "KNN during training": knn_dict,
                "IMG_RESIZE": IMG_RESIZE,
                "Embeddings during training": embed_dict,
            }

            with open(f'{folder}/model_details/{MODEL_FILENAME}_details.pkl', 'wb') as f:
                pickle.dump(model_details, f)
                
            print(f"Model details saved to {MODEL_FILENAME}_details.pkl")


            ###################### EVALUATION #########################
            eval_dict = model_eval(
                model,
                data_train,
                data_test,
                data_loader_classifier,
                n_classes=N_CLASSES,
            )

            with open(f'{folder}/model_eval/{MODEL_FILENAME}_eval.pkl', 'wb') as f:
                pickle.dump(eval_dict, f)

            print(f"Evaluation results saved to {MODEL_FILENAME}_eval.pkl")

            

        # Mixed eqCLR (2 eq layer)

        for N in [2, 4, 8]:

            MODEL_FILENAME = f"C-0003_seed{seed}_{name_data}-mixed_eqCLR_resnet18_2eqblocks_N{N}_{group}_kp_1000epochs_lr{BASE_LR}"
            print(f"Model filename: {MODEL_FILENAME}")

            ##################### NETWORK ARCHITECTURE #########################
            if group == 'flip_rot':
                model = EqResNet(N=int(N/2), r2_act=gspaces.flipRot2dOnR2(int(N/2)), in_channels=IN_CHANNELS, eq_blocks=2, maxpool=MAXPOOL)
            elif group == 'rot':
                model = EqResNet(N=N, in_channels=IN_CHANNELS, eq_blocks=2, maxpool=MAXPOOL)

            optimizer = SGD(
                model.parameters(),
                lr=BASE_LR * BATCH_SIZE / 256,
                momentum=MOMENTUM,
                weight_decay=WEIGHT_DECAY,
            )

            scheduler = CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

            ##################### TRAINING LOOP #########################

            print("Starting training.")

            device = "cuda"

            model.to(device)
            model.train()
            knn_dict = {}
            embed_dict = {}
            training_start_time = time.time()
            time_train = 0.0

            for epoch in range(N_EPOCHS):
                epoch_loss = 0.0
                start_time = time.time()

                for batch_idx, batch in enumerate(data_loader_ssl):
                    views, _ = batch

                    views = [view.to(device, non_blocking=True) for view in views]

                    optimizer.zero_grad()

                    _, z1 = model(views[0])
                    _, z2 = model(views[1])
                    loss = infoNCE(torch.cat((z1, z2)))
                    epoch_loss += loss.item()

                    loss.backward()
                    optimizer.step()

                end_time = time.time()
                time_train += end_time - start_time

                scheduler.step()

                if EVAL_DURING_TRAIN and should_eval(epoch, n_epochs=N_EPOCHS):
                    model.eval()
                    with torch.no_grad():
                        X_train, y_train, Z_train = dataset_to_X_y(data_train, model)
                        X_test, y_test, Z_test = dataset_to_X_y(data_val, model)

                        knn_acc = eval_knn_single(X_train, y_train, X_test, y_test)
                        knn_dict[epoch] = knn_acc

                        if ITER_SAVE_EMBED is not None and epoch % ITER_SAVE_EMBED == 0:
                            embed_dict[epoch] = {
                                "X_test": X_test,
                                "y_test": y_test,
                            }
                    model.train()

                if epoch % PRINT_EVERY_EPOCHS == 0:
                    print(
                        f"Epoch {epoch + 1}, "
                        f"average loss {epoch_loss / len(data_loader_ssl):.4f}, "
                        f"{end_time - start_time:.1f} s",
                        f"KNN accuracy {knn_dict.get(epoch, 'N/A')}",
                        flush=True
                    )

            training_end_time = time.time()
            hours = (training_end_time - training_start_time) / 60 // 60
            minutes = (training_end_time - training_start_time) / 60 % 60
            average = (training_end_time - training_start_time) / N_EPOCHS
            print(
                f"Total training length for {N_EPOCHS} epochs: {hours:.0f}h {minutes:.0f}min",
                f"({average:.1f} sec/epoch)",
                flush=True
            )

            model.eval()
            torch.save(model.state_dict(), f'{folder}/model_weights/{MODEL_FILENAME}_weights.pt')
            print(f"Model saved to {MODEL_FILENAME}_weights.pt")

            model_details = {
                "Filename": MODEL_FILENAME,
                "Model structure": str(model),
                "N_EPOCHS": N_EPOCHS,
                "BATCH_SIZE": BATCH_SIZE,
                "BASE_LR": BASE_LR,
                "WEIGHT_DECAY": WEIGHT_DECAY,
                "MOMENTUM": MOMENTUM,
                "CROP_LOW_SCALE": CROP_LOW_SCALE,
                "GRAYSCALE_PROB": GRAYSCALE_PROB,
                "PROJECTOR_HIDDEN_SIZE": PROJECTOR_HIDDEN_SIZE,
                "PROJECTOR_OUTPUT_SIZE": PROJECTOR_OUTPUT_SIZE,
                "Training augmentations": transforms_ssl,
                "Training time": training_end_time - training_start_time,
                "Training time w/o evaluation": time_train,
                "MAXPOOL": MAXPOOL,
                "KNN during training": knn_dict,
                "IMG_RESIZE": IMG_RESIZE,
                "Embeddings during training": embed_dict,
            }

            with open(f'{folder}/model_details/{MODEL_FILENAME}_details.pkl', 'wb') as f:
                pickle.dump(model_details, f)
                
            print(f"Model details saved to {MODEL_FILENAME}_details.pkl")


            ##################### EVALUATION #########################
            eval_dict = model_eval(
                model,
                data_train,
                data_test,
                data_loader_classifier,
                n_classes=N_CLASSES,
            )

            with open(f'{folder}/model_eval/{MODEL_FILENAME}_eval.pkl', 'wb') as f:
                pickle.dump(eval_dict, f)

            print(f"Evaluation results saved to {MODEL_FILENAME}_eval.pkl")

