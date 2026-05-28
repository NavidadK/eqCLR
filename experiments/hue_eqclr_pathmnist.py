# eqCLR_pmnist
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import Dataset, DataLoader
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

import torchvision.transforms as transforms
from torchvision.models import resnet18

import numpy as np
import time
import pickle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from medmnist import PathMNIST

from eqCLR.eq_resnet import EqResNet18, EqResNet, EqResNet_hue, eq_resnet, EqResNet18_hue, EqResNet18_hue2
from eqCLR.test_resnet import Wide_ResNet
from evaluation import model_eval, eval_knn_single, dataset_to_X_y, lin_eval_rep
from escnn import gspaces
###################### PARAMS ##############################

folder = 'results_hue/pathmnist'

BATCH_SIZE = 512
N_EPOCHS = 100 # 1000
N_CPU_WORKERS = 16
BASE_LR = 0.06         # 0.03
WEIGHT_DECAY = 5e-4    # important
MOMENTUM = 0.9
PROJECTOR_HIDDEN_SIZE = 1024
PROJECTOR_OUTPUT_SIZE = 128
CROP_LOW_SCALE = 0.2
GRAYSCALE_PROB = 0.1   # important
PRINT_EVERY_EPOCHS = 1
EVAL_DURING_TRAIN = True
ITER_SAVE_EMBED = 50
IMG_RESIZE = 33  # if None, use original size 28x28
MAXPOOL = True  # 'avg' or 'max' or None


for rs in range(3):
    np.random.seed(rs)
    torch.manual_seed(rs)

    MODEL_FILENAME = f"0002-seed{rs}_pathmnist_hue_eqCLR_N8"

    print(f"Model filename: {MODEL_FILENAME}")

    ###################### DATA LOADER #########################
    class RGBToHSV:
        def __init__(self):
            pass
        def __call__(self, img_tensor):
            # img_tensor: [C,H,W] in [0,1]
            img_np = img_tensor.permute(1, 2, 0).numpy()  # [H,W,C]
            hsv_np = rgb_to_hsv(img_np)                    # [H,W,C], hue in [0,1]
            hsv_tensor = torch.tensor(hsv_np).permute(2, 0, 1)  # back to [C,H,W]
            return hsv_tensor
        
    if IMG_RESIZE is not None:
        transform = transforms.Compose([
            transforms.Resize((IMG_RESIZE, IMG_RESIZE)),
            transforms.ToTensor(),
            RGBToHSV()
        ])
    else:
        transform = transforms.ToTensor()

    transform_test = transforms.Compose([
        transforms.Resize((IMG_RESIZE, IMG_RESIZE)),
        transforms.ToTensor(),
        RGBToHSV()
    ])

    pmnist_train = PathMNIST(split='train', download=False, size=28, root='data/pathmnist/', transform=transform)
    pmnist_test = PathMNIST(split='test', download=False, size=28, root='data/pathmnist/', transform=transform_test)
    pmnist_val = PathMNIST(split='val', download=False, size=28, root='data/pathmnist/', transform=transform_test)
    print("Data loaded.")

    if IMG_RESIZE is None:
        IMG_RESIZE = pmnist_train[0][0].shape[1]  # get image size from dataset
    print(f"Image size (resized): {IMG_RESIZE}")

    # additional rotation
    class RandomRightAngleRotation:
        """Randomly rotate PIL image by 90, 180, or 270 degrees."""
        def __call__(self, x):
            angle = int(torch.randint(1, 4, ()).item()) * 90
            return x.rotate(angle)

    transforms_ssl = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=IMG_RESIZE, scale=(CROP_LOW_SCALE, 1)),
            RandomRightAngleRotation(), # additional rotation
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
            ),
            # transforms.RandomGrayscale(p=GRAYSCALE_PROB),
            transforms.ToTensor(), # NB: runtime faster when this line is last
            RGBToHSV()
        ]
    )

    class PairedTransform:
        def __init__(self, transform):
            self.transform = transform

        def __call__(self, x):
            return (self.transform(x), self.transform(x))

    paired_ssl_transforms = PairedTransform(transforms_ssl)

    pmnist_train_ssl = PathMNIST(split='train', download=False, size=28, root='data/pathmnist/', transform=paired_ssl_transforms)

    pmnist_loader_ssl = DataLoader(
        pmnist_train_ssl,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=N_CPU_WORKERS,
        pin_memory=True,
    )

    ###################### NETWORK ARCHITECTURE #########################

    def infoNCE(features, temperature=0.5):
        x = F.normalize(features)
        cos_xx = x @ x.T / temperature
        cos_xx.fill_diagonal_(float("-inf"))
        
        batch_size = cos_xx.size(0) // 2
        targets = torch.arange(batch_size * 2, dtype=int, device=cos_xx.device)
        targets[:batch_size] += batch_size
        targets[batch_size:] -= batch_size

        return F.cross_entropy(cos_xx, targets)

    # fully equivariant
    model = EqResNet_hue(N=4, eq_blocks=4, maxpool=MAXPOOL, projector_hidden_size=PROJECTOR_HIDDEN_SIZE, n_classes=PROJECTOR_OUTPUT_SIZE, adjust_channels='keep_param')
    # mixed equivariant
    # model = EqResNet_hue(N=4, eq_blocks=2, maxpool=MAXPOOL, projector_hidden_size=PROJECTOR_HIDDEN_SIZE, n_classes=PROJECTOR_OUTPUT_SIZE, adjust_channels='keep_param')

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

        for batch_idx, batch in enumerate(pmnist_loader_ssl):
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

        if epoch in [0, 5, 15, 30, 50, 75, 99]:
            model.eval()
            with torch.no_grad():
                X_train, y_train, Z_train = dataset_to_X_y(pmnist_val, model)
                X_test, y_test, Z_test = dataset_to_X_y(pmnist_test, model)

                knn_acc = eval_knn_single(X_train, y_train, X_test, y_test)
                knn_dict[epoch] = knn_acc

                if ITER_SAVE_EMBED is not None and (epoch + 1) % ITER_SAVE_EMBED == 0:
                    embed_dict[epoch] = {
                        "X_test": X_test,
                        "y_test": y_test,
                    }
            model.train()

        if (epoch + 1) % PRINT_EVERY_EPOCHS == 0:
            print(
                f"Epoch {epoch + 1}, "
                f"average loss {epoch_loss / len(pmnist_loader_ssl):.4f}, "
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

    ###################### EVALUATION #########################
    transforms_classifier = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=IMG_RESIZE, scale=(CROP_LOW_SCALE, 1)),
            transforms.RandomHorizontalFlip(),
            RandomRightAngleRotation(), # additional rotation
            transforms.ToTensor(),
            RGBToHSV()
        ]
    )

    pmnist_train_classifier = PathMNIST(split='train', download=False, size=28, root='data/pathmnist/', transform=transforms_classifier)

    pmnist_loader_classifier = DataLoader(
        pmnist_train_classifier,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=N_CPU_WORKERS,
    )

    eval_dict = model_eval(
        model,
        pmnist_train,
        pmnist_test,
        pmnist_loader_classifier,
        n_classes=9,
    )

    with open(f'{folder}/model_eval/{MODEL_FILENAME}_eval.pkl', 'wb') as f:
        pickle.dump(eval_dict, f)

    print(f"Evaluation results saved to {MODEL_FILENAME}_eval.pkl")
