# evaluation.py
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
from sklearn.metrics import roc_auc_score, roc_curve

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dataset_to_X_y(dataset, model, device=device):
    X = []
    y = []
    Z = []

    for batch_idx, batch in enumerate(DataLoader(dataset, batch_size=1024)):
        images, labels = batch

        h, z = model(images.to(device))

        X.append(h.cpu().numpy())
        Z.append(z.cpu().numpy())
        y.append(labels.cpu().numpy().ravel())

    X = np.vstack(X)
    Z = np.vstack(Z)
    y = np.hstack(y)

    return X, y, Z

import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F


def eval_knn(X_train, y_train, X_test, y_test):
    eval_dict = {}

    for metric in ["euclidean", "cosine"]:
        print(f'---------- KNN with {metric} distance ----------')
        eval_dict[metric] = {}

        for k in [1, 5, 10]:
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric, n_jobs=-1)
            knn.fit(X_train, y_train)
            acc = knn.score(X_test, y_test)
            eval_dict[metric][k] = acc
            print(f"KNN (k={k}, metric={metric}): {acc*100:.2f}%")

    return eval_dict

def eval_knn_single(X_train, y_train, X_test, y_test, metric='euclidean', n_neighbors=10):

    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, n_jobs=-1)
    knn.fit(X_train, y_train)
    acc = knn.score(X_test, y_test)

    return acc
    
def eval_logreg(X_train, y_train, X_test, y_test, binary=False):
    # Logistic Regression
    logreg = LogisticRegression(max_iter=1000, n_jobs=-1)
    logreg.fit(X_train, y_train)
    log_reg = logreg.score(X_test, y_test)
    print(f"Logistic Regression: {log_reg*100:.2f}%")

    lin = LogisticRegression(penalty=None, solver="saga")
    lin.fit(X_train, y_train)
    lin_acc = lin.score(X_test, y_test)
    print(f"Linear accuracy (sklearn): {lin_acc}", flush=True)

    if binary:
        y_score = logreg.decision_function(X_test)
        auc = roc_auc_score(y_test, y_score)
        print(f"ROC AUC: {auc:.4f}")

        return log_reg, lin_acc, auc
    else:
        return log_reg, lin_acc

def lin_eval_rep(X_train, y_train, X_test, y_test, n_classes, n_epochs=100, adam_lr=0.01, device=device, return_model=False):

    X_train = torch.tensor(X_train, device=device)
    X_test = torch.tensor(X_test, device=device)
    y_train = torch.tensor(y_train, device=device)
    y_test = torch.tensor(y_test, device=device)

    classifier = nn.Linear(X_train.shape[1], n_classes)
    classifier.to(device)
    classifier.train()

    optimizer = Adam(classifier.parameters(), lr=adam_lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    for epoch in range(n_epochs):
        N = len(X_train)
        perm = torch.randperm(N)
        perm = perm[:N - (N % 1000)]              # drop remainder
        batches = perm.view(-1, 1000)        
        for idx in batches:
            optimizer.zero_grad()
            logits = classifier(X_train[idx])
            loss = F.cross_entropy(logits, y_train[idx])
            loss.backward()
            optimizer.step()
        scheduler.step()

    classifier.eval()
    with torch.no_grad():
        yhat = classifier(X_test)

    acc = (yhat.argmax(axis=1) == y_test).cpu().numpy().mean()
    print(f"Linear accuracy (Adam on precomputed representations): {acc}", flush=True)

    if n_classes == 2:
        # Get probabilities for class 1
        probs = torch.softmax(yhat, dim=1)[:, 1].cpu().numpy()
        y_true = y_test.cpu().numpy()
        auc = roc_auc_score(y_true, probs)
        fpr, tpr, thresholds = roc_curve(y_true, probs)
        print(f"Linear ROC-AUC: {auc:.4f}", flush=True)
        if return_model:
            return acc, auc, classifier
        else:            
            return acc, auc

    if return_model:
        return acc, classifier
    else:
        return acc
        
def lin_eval_aug(test_data, loader_classifier, model, n_classes, dim_represenations, n_epochs=100, adam_lr=0.01, adam_wd=5e-6, print_every_epochs=5, device=device, return_model=False):
    classifier = nn.Linear(dim_represenations, n_classes)
    # for param in model.backbone.parameters():
    #     param.requires_grad = False
    for name, module in model.named_children():
        if name != "fully_net" and name != "projector":
            for param in module.parameters():
                param.requires_grad = False

    optimizer = Adam(classifier.parameters(), lr=adam_lr, weight_decay=adam_wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    classifier.to(device)
    classifier.train()
    training_start_time = time.time()

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        start_time = time.time()

        for batch_idx, batch in enumerate(loader_classifier):
            view, y = batch

            optimizer.zero_grad()

            h, _ = model(view.to(device))
            h = h.detach() 
            logits = classifier(h)
            loss = F.cross_entropy(logits, y.to(device).squeeze().long())
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

        end_time = time.time()
        if (epoch + 1) % print_every_epochs == 0:
            print(
                f"Epoch {epoch + 1}, "
                f"average loss {epoch_loss / len(loader_classifier):.4f}, "
                f"{end_time - start_time:.1f} s",
                flush=True
            )

    scheduler.step()

    training_end_time = time.time()
    hours = (training_end_time - training_start_time) / 60 // 60
    minutes = (training_end_time - training_start_time) / 60 % 60
    print(
        f"Total classifier training length for {n_epochs} epochs: {hours:.0f}h {minutes:.0f}min",
        flush=True
    )

    classifier.eval()
    with torch.no_grad():
        yhat = []
        y = []

        for batch_idx, batch in enumerate(DataLoader(test_data, batch_size=1024)):
            images, labels = batch

            h, _ = model(images.to(device))
            logits = classifier(h)

            yhat.append(logits.cpu().numpy())
            y.append(labels.cpu().numpy().ravel())

        yhat = np.vstack(yhat)
        y = np.hstack(y)

    acc = (yhat.argmax(axis=1) == y).mean()
    print('acc', acc)
    print(f"Linear accuracy (trained with augmentations): {acc}", flush=True)

    if return_model:
        return acc, classifier
    else:
        return acc


def model_eval(model, data_train, data_test, loader_classifier, n_classes, eval_aug=True):

    with torch.no_grad():
        X_train, y_train, Z_train = dataset_to_X_y(data_train, model)
        X_test, y_test, Z_test = dataset_to_X_y(data_test, model)

    # KNN Evaluation
    print('------------------ KNN Evaluation ------------------')
    eval_dict = eval_knn(X_train, y_train, X_test, y_test)

    # Logistic Regression
    print('------------------ Logistic Regression ------------------')
    log_reg, lin_acc = eval_logreg(X_train, y_train, X_test, y_test)
    eval_dict["logistic_regression"] = log_reg
    eval_dict["linear_accuracy"] = lin_acc

    # Linear Evaluation with precomputed representations
    print('------------------ Linear Evaluation with precomputed representations ------------------')
    lin_acc_rep = lin_eval_rep(X_train, y_train, X_test, y_test, n_classes)
    eval_dict["linear_accuracy_rep"] = lin_acc_rep

    # Linear Evaluation with augmentations
    if eval_aug:
        print('------------------ Linear Evaluation with augmentations ------------------')
        lin_acc_aug = lin_eval_aug(data_test, loader_classifier, model, n_classes, dim_represenations=X_train.shape[1])
        eval_dict["linear_accuracy_aug"] = lin_acc_aug

    return eval_dict

def should_eval(epoch, n_epochs=1000):
    if epoch < 10:
        return True          # every epoch
    elif epoch < 50:
        return epoch % 2 == 0
    elif epoch < 100:
        return epoch % 5 == 0
    elif epoch < 150:
        return epoch % 10 == 0
    elif epoch < 300:
        return epoch % 15 == 0
    elif epoch < 500:
        return epoch % 20 == 0
    elif epoch == n_epochs - 1:
        return True
    else:
        return epoch % 25 == 0