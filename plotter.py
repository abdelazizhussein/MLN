import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import click
import sklearn.metrics as metrics
from sklearn.metrics import balanced_accuracy_score


def score_distribution(true_label,nn_label,training_directory):
    """
    Plots the output score distribution of ghosts and real tracks (training datasets)
    """
    x = np.linspace(0,1,168)
    plt.figure()
    plt.xlabel('Score')
    plt.ylabel('Number of events')
    
    ghost_mask = true_label == 1
    true_mask = true_label == 0
    nn_label = nn_label.detach().numpy()
    plt.hist(nn_label[true_mask],histtype="step",label = 'Real Tracks')
    plt.hist(nn_label[ghost_mask],histtype="step",label='Ghost Tracks')
    plt.legend()
    plt.savefig(os.path.join(training_directory,'plots/training_score_dist.png'))
    
def plot_loss(loss,epochs,training_directory):
    """
    Plots loss vs epoch
    """
    x = np.arange(0,epochs,1)
    plt.figure()
    plt.xlabel('Epoch Time')
    plt.ylabel('loss (A.U.)')
    plt.plot(x,loss)
    plt.savefig(os.path.join(training_directory,'plots/loss.png'))
    return

def plot_ROC(y_test,y_pred,training_directory):
    """
    Plots ROC curve
    """
    y_test = y_test.values
    y_pred = y_pred.detach().numpy()
    fpr, tpr, threshold = metrics.roc_curve(y_test,y_pred)
    plt.figure()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot(fpr,tpr)
    plt.savefig(os.path.join(training_directory,'plots/roc.png'))
    return

    # def plot_features(X_train,true_label):
    #     for (feature_name, data) in X_train.iteritems():
    #         plt.figure()
    #         plt.plot(data.values)
    #         plt.xlabel('Score')
    #         plt.ylabel('Number of events')
    #         ghost_mask = true_label== 1
    #         true_mask = true_label == 0
    #         print(true_label.values)
    #         plt.hist(data[true_mask],histtype="step",label = 'Real Tracks',density=True)
    #         plt.hist(data[ghost_mask],histtype="step",label = 'Ghost Tracks',density=True)
    #         plt.savefig(feature_name+".png")

    # plot_features(dataset,labels)

from IPython import embed
@click.command()
@click.option('-t','--training-directory',required=True, help="Training directory")
def plotter(training_directory):
    """calls funcs to make different plots"""
    #load saved data from training
    with open(os.path.join(training_directory,'objs.pkl'), 'rb') as f: 
        EPOCHS,loss,y_train,y_robust,x_val,y_val= pickle.load(f)

    #make plots directory
    if not os.path.exists(os.path.join(training_directory,'plots')):
        os.makedirs(os.path.join(training_directory,'plots'))

    #make plots
    plot_loss(loss,EPOCHS,training_directory)
    score_distribution(y_train,y_robust,training_directory)
    plot_ROC(y_train,y_robust,training_directory)



if __name__ == '__main__':
    plotter()