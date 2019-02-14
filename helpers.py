import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_features(dataset,dx,dy):
    plt.figure(figsize = (dx,dy))
    gs1 = gridspec.GridSpec(6, 5)
    gs1.update(wspace=0.025, hspace=0.4)
    bins=30

    column_names = dataset.columns.values[:-1].astype(str)

    for i,name in enumerate(column_names):

        ax = plt.subplot(gs1[i])

        min_val = np.percentile(dataset[name].values,0.1)
        max_val = np.percentile(dataset[name].values,99.9)

        class1_vals = dataset.loc[dataset['Class']==1][name].values
        class0_vals = dataset.loc[dataset['Class']==0][name].values

        ax.hist(class0_vals,bins=bins,
                 normed=True,range=[min_val,max_val],
                 histtype='step',color='black',label='normal')
        ax.hist(class1_vals,bins=bins,
                 normed=True,range=[min_val,max_val],
                 histtype='step',color='red',label='fraud')

        ax.get_yaxis().set_visible(False)
        ax.set_title(name)

    plt.show()


def train_test_split(dataset, train_frac=0.7):
    
    #Shuffle
    dataset = dataset.sample(frac=1)
    
    #Get the classes separately
    class0 = dataset.loc[dataset['Class']==0]
    class1 = dataset.loc[dataset['Class']==1]
    
    #Get the limit index between train and test
    train0_lim = int(class0.shape[0]*train_frac)
    train1_lim = int(class1.shape[0]*train_frac) 
    
    #Class 0 split
    train0 = class0.values[:train0_lim]   
    test0 = class0.values[train0_lim:]    

    #Class 1 split
    train1 = class1.values[:train1_lim]   
    test1 = class1.values[train1_lim:]        
    
    #Stick them back together and shuffle
    train = np.vstack([train0,train1])
    test  = np.vstack([test0,test1])    
    np.random.shuffle(train)
    np.random.shuffle(test)
    
    np.save('data/train.npy',train)
    np.save('data/test.npy',test)

    return train,test
    
def evaluate_model(test_prob,y_test,dx=10,dy=4.5,fontsize=14):

    fig, ax_array = plt.subplots(1,2)
    fig.set_size_inches((dx, dy))

    #Left: the score distribution
    inds_0 = np.where(y_test == 0)[0]
    inds_1 = np.where(y_test == 1)[0]

    bins=40
    ax_array[0].hist(test_prob[inds_0],
             bins=bins,normed=True,color='blue',range=[0,1],
             histtype='step',label='Normal')
    ax_array[0].hist(test_prob[inds_1],
             bins=bins,normed=True,color='red',range=[0,1],
             histtype='step',label='Fraudulent')

    ax_array[0].legend(loc=0,fontsize=fontsize)
    ax_array[0].set_xlabel('Probability Fraudulent',fontsize=fontsize)

    #Right: the ROC curve
    fpr,tpr,_ = sklearn.metrics.roc_curve(y_true=y_test,y_score=test_prob)

    auc = sklearn.metrics.roc_auc_score(y_true=y_test,y_score=test_prob)

    label = "AUROC = %1.3f"%auc
    ax_array[1].plot(fpr,tpr,color='black',label=label)
    delta=0.025
    ax_array[1].set_xlim(0-delta,1)
    ax_array[1].set_ylim(0,1+delta)
    ax_array[1].set_xlabel("False Positive Rate",fontsize=fontsize)
    ax_array[1].set_ylabel("True Positive Rate",fontsize=fontsize)
    ax_array[1].legend(loc=4,fontsize=fontsize)

    plt.show()
