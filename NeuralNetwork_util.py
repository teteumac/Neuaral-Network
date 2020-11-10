import pandas as pd
import numpy as  np

import keras.backend as K
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn import metrics
from sklearn.metrics import roc_curve,precision_recall_curve

def f1_score(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

def do_scaler(X,Y, scaler_X, scaler_Y):
    if scaler_X == None:
        X_scaled = X
    else:
        X_scaled = scaler_X.fit_transform(X)
    if scaler_Y == None:
        Y_scaled = Y
    else:
        Y_scaled = scaler_Y.fit_transform(Y)
    return X_scaled, Y_scaled, scaler_X, scaler_Y

def test_model(X,Y, model, scaler_X = None, verbose = 1):
    if scaler_X is not None:
        X_scaled = scaler_X.transform(X)
    else:
        X_scaled = X
    y_probs = model.predict_proba(X_scaled)  # calculate the probability
    preds = model.predict(X_scaled)
    prec, rec, thresh = precision_recall_curve(Y, y_probs[:,1])
    bidx = np.argmax(prec * rec)
    best_cut = thresh[bidx]
    print(best_cut)

    preds = y_probs[:,1] >= best_cut

    if verbose == 1:

        fpr, tpr, thresholds = metrics.roc_curve(Y, y_probs[:,1], drop_intermediate=False)
        print(f'AUC = {metrics.auc(fpr, tpr)}')
        print("Purity in test sample     : {:2.2f}%".format(100 * precision_score(Y, preds)))
        print("Efficiency in test sample : {:2.2f}%".format(100 * recall_score(Y, preds)))
        print("Accuracy in test sample   : {:2.2f}%".format(100 * accuracy_score(Y, preds)))