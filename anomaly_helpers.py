# Import necessary libraries
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest

# Filepaths
data_folder_path="data"


###
### CREATING THE DATASET
###

def ask_for_train_size(max_length=10000):
    train_size = None
    print("Number of training samples to be used:")
    print("Please choose a number between 100 and {max}".format(max=max_length))
    while train_size is None:
        input_value = input(" ")
        try:
            # try and convert the string input to a number
            train_size = int(input_value)
            if not 100<=train_size<=max_length:
                train_size = None
                print("Please choose a number between 100 and {max}".format(max=max_length))
        except ValueError:
            print("Please enter a number:")
    print("")
    return train_size
     
def ask_for_train_contamination():
    train_contam = None
    print("Portion of training samples to be attacks:")
    print("Please choose a number between 0 and 0.5 .")
    while train_contam is None:
        input_value = input(" ")
        try:
            # try and convert the string input to a float number
            train_contam = float(input_value)
            if not 0<=train_contam<=0.5:
                train_contam = None
                print("Please choose a number between 0 and 0.5 .")
        except ValueError:
            print("Please enter a number:")
    print("")
    return train_contam


def generate_training_data(data_full, labels_full, 
                       training_samples=10000, 
                       training_contamination=0.25):
    X=data_full
    y=labels_full
    training_samples=np.minimum(training_samples, y.shape[0])

    # Calculate subset sizes
    attack_filter=y.attack==1
    attack_filter.sum()
    n_attack = np.ceil(training_samples * training_contamination).astype('int')
    if n_attack>attack_filter.sum():
        # reduce number of attacks and fill with normal
        print("warning_msg")
        n_attack = attack_filter.sum()
    n_normal = training_samples - n_attack

    # Create subframes
    X_normal=X.loc[~attack_filter, :]
    y_normal=y.loc[~attack_filter, :]
    X_attack=X.loc[attack_filter, :]
    y_attack=y.loc[attack_filter, :]

    # Sample normal and attacks
    if n_attack>0:
        _, X_n, _, y_n = train_test_split(X_normal, y_normal, 
                                          test_size=n_normal, random_state=7)
        _, X_a, _, y_a = train_test_split(X_attack, y_attack, 
                                          test_size=n_attack, random_state=7)
        data_train = pd.concat([X_n, X_a])
        labels_train = pd.concat([y_n, y_a])
    else:
        _, data_train, _, labels_train = train_test_split(X_normal, y_normal, 
                                          test_size=n_normal, random_state=42)
    
    return data_train, labels_train


def label_counter(labels):
    label_counter = labels.attack_detail.value_counts().to_frame().rename(columns={"attack_detail": "frequency"})
    label_counter.index.set_names(['class'], inplace=True)
    return label_counter

def create_dataset():
    ### data_folder_path="data"
    # import data
    data_full=pd.read_csv(os.path.join(data_folder_path, "data_AMLD.csv"))
    labels_full=pd.read_csv(os.path.join(data_folder_path, "labels_AMLD.csv"))

    # get parameters from user
    training_samples = ask_for_train_size(max_length=data_full.shape[0])
    training_contamination = ask_for_train_contamination()
    # generate training data
    data_train, labels_train = generate_training_data(data_full, labels_full, 
                                                      training_samples=training_samples, 
                                                      training_contamination=training_contamination)
    # display training data info
    print("Your training set has the following composition:")
    display(label_counter(labels_train))
    
    # import test datasets
    data_10=pd.read_csv(os.path.join(data_folder_path, "data_test_10K.csv"))
    labels_10=pd.read_csv(os.path.join(data_folder_path, "labels_test_10K.csv"))
    data_1=pd.read_csv(os.path.join(data_folder_path, "data_test_1K.csv"))
    labels_1=pd.read_csv(os.path.join(data_folder_path, "labels_test_1K.csv"))
    # bundle datasets
    dataset={"data_train": data_train,
             "labels_train": labels_train,
             "data_test1": data_1,
             "labels_test1": labels_1,
             "data_test10": data_10,
             "labels_test10": labels_10}
    return dataset


###
### VISUALISING THE DATA
###


def downsample_for_plot(data, labels, size):
    _, X, _, labels_plot = train_test_split(data, labels, test_size = size, 
                            random_state=42, stratify=labels.attack_detail)
    return X, labels_plot

# Build visualiser of data sets using TSNE
def TSNE_visualiser(X, labels_plot, colour_by_level=2, marker_list=[], perplexity=30):
    tSNE = TSNE(n_components=2, learning_rate=300,
            perplexity=perplexity, early_exaggeration=12,
            init='random', random_state=42)    
    
    X.reset_index(drop=True, inplace=True)
    labels_plot.reset_index(drop=True, inplace=True)
    
    scaler=StandardScaler()
    X_scaled=scaler.fit_transform(X)

    pca = PCA(n_components=None)
    X_pca=pca.fit_transform(X_scaled)
    
    X_tSNE = tSNE.fit_transform(X_pca)
    X_tSNE = pd.DataFrame(data=X_tSNE)
    
    colouring = labels_plot.iloc[:, colour_by_level]
    
    color_dict = dict({'normal':'lightblue',
                       'neptune':'green',
                       'smurf':'yellowgreen',
                       'teardrop': 'chartreuse',
                       'back': 'limegreen',
                       'pod': 'olivedrab',
                       'satan': 'red',
                       'ipsweep': 'darkred',
                       'portsweep': 'salmon',
                       'nmap':'orange',
                       'warezclient': 'darkorchid',
                       'guess_passwd': 'indigo'})
    
    
    if marker_list==[]:
        marker_list=[0, 1]*labels_plot.shape[0]
        marker_list=marker_list[:labels_plot.shape[0]]
    if len(marker_list)!=labels_plot.shape[0]:
        print("marker_list and data length mismatch, abort")
        return
    
    plt.figure(figsize=(12,12))
    sns.scatterplot(x=X_tSNE[0], y=X_tSNE[1], hue=colouring, 
                    s=25, palette=color_dict)
    plt.legend()
    plt.show()


def visualise_training_data(dataset, max_plot=1000):
    print("Plotting", max_plot, "samples of the training data")
    
    data = dataset["data_train"]
    labels = dataset["labels_train"]
    if data.shape[0] > max_plot:
        data, labels = downsample_for_plot(data,
                                           labels, 
                                           size=max_plot)  
    TSNE_visualiser(data, labels, colour_by_level=2, perplexity=30)

def visualise_test_data(dataset, max_plot=1000):
    print("Plotting", max_plot, "samples of the test data")
    
    data = dataset["data_test1"]
    labels = dataset["labels_test1"]
    if data.shape[0] > max_plot:
        data, labels = downsample_for_plot(data,
                                           labels, 
                                           size=1000)
    TSNE_visualiser(data, labels, colour_by_level=2, perplexity=30)

    
###
### TRAINING AND EVALUATIING THE ANOMALY DETECTOR ON KDD DATA
###


def kdd_train_predict_clf(X_train, X_test, labels_test, clf):
    # train model
    clf.fit(X_train)
    # make predictions on test set
    y_pred_clf = clf.predict(X_test)
    # in y_pred_clf outlier = -1 and normal = 0
    y_pred=pd.Series((y_pred_clf<0)*1)
    # create data frame with labels and predictions
    y_eval_df = labels_test.copy()
    y_eval_df['pred']= y_pred
    return y_eval_df    

def kdd_classification_report(y_eval_df):
    """
    Calculate and plot confusion matrix from y_eval_df
    Report of key counts
    """
    # extract ground truth and predictions
    y_true =y_eval_df.attack
    y_pred=y_eval_df.pred
    cm = confusion_matrix(y_true, y_pred)
    # Summary
    print("Attacks identified correctly (true positive):", cm[1,1], " of ", cm[1,1]+ cm[1,0], "(", (100*cm[1,1]/(cm[1,1]+ cm[1,0])).round(1), "%)")
    print("Attacks missed (false negative):", cm[1,0], " of ", cm[1,1]+ cm[1,0], "(", (100*cm[1,0]/(cm[1,1]+ cm[1,0])).round(1), "%)")
    print("False alarms (false positive):", cm[0,1], " of", cm[0,0]+ cm[0,1], "(", (100*cm[0,1]/(cm[0,0]+ cm[0,1])).round(1), "%)")
    print("Total number of misclassified samples:", cm[0,1]+cm[1,0])
    # create and plot confusion_matrix
    plot_labels = ['Normal', 'Attack']
    plt.figure(figsize = (6, 5))
    heatmap=sns.heatmap(cm, annot=True, fmt = 'd')  # fmt = 'd' suppresses scientific notation
    heatmap.set(xlabel="Predicted", ylabel="True", 
                xticklabels=plot_labels, yticklabels=plot_labels)
    plt.show()

def kdd_classification_report_detailed(y_eval_df):
    # filter for all misclassified data points
    misclassified=y_eval_df[y_eval_df['attack']!=y_eval_df['pred']]
    missed=misclassified[misclassified['pred']==0]
    # build reporting data frame with originally present types, number of misclassified per type and the percentage
    df1=pd.DataFrame(y_eval_df['attack_detail'].value_counts()).reset_index()
    df2=pd.DataFrame(missed['attack_detail'].value_counts()).reset_index()
    df=df1.merge(df2, how='outer', on='index').rename({'index':'attack_detail', 'attack_detail_x': 'present', 'attack_detail_y': 'missed'}, axis=1)
    df=df.fillna(0)
    df.set_index('attack_detail', inplace=True)
    # Correct missed normals from 0 to false alarms
    false_alarms=confusion_matrix(y_eval_df.attack, y_eval_df.pred)[0,1]
    df.loc['normal', 'missed']=false_alarms
    df.missed=df.missed.astype(int)
    df['pct_missed']=round(100*df.missed/df.present, 1)
    return df

def draw_decision_function(X_test, y_test, clf, n_bins=200):
    scoring =  clf.decision_function(X_test) 
    plot_min=1.1*np.min(scoring)
    plot_max=1.1*np.max(scoring)
    bins = np.linspace(plot_min, plot_max, n_bins)

    plt.figure(figsize=(12,6))
    plt.hist(scoring[y_test.attack == 0], bins, alpha=0.5, label='True normal')
    plt.hist(scoring[y_test.attack == 1], bins, alpha=0.5, label='True outlier')
    ### TODO
    ### ADD red vertical line at 0
    plt.axvline(x=0, ls='--', color='r', label="Decision boundary")
    plt.title('Outlier scores for test samples (below 0 assigned as outlier)')
    plt.xlim(left=plot_min, right=plot_max)
    plt.ylim(bottom=0)
    plt.legend()
    plt.show()
    
def kdd_train_evaluate_report_clf(X_train, X_test, y_test, clf):
    y_eval_df=kdd_train_predict_clf(X_train, X_test, y_test, clf)
    draw_decision_function(X_test, y_test, clf)
    print("Summary report:")
    kdd_classification_report(y_eval_df)
    print("Detailed report:")
    display(kdd_classification_report_detailed(y_eval_df))
    

def visualise_test_data_detailed(X_test, y_eval_df, max_plot=1000):
    print("Plotting", max_plot, "samples of the test data")
    
    data = X_test
    labels_plot = y_eval_df
    if data.shape[0] > max_plot:
        data, labels_plot = downsample_for_plot(data,
                                           labels, 
                                           size=1000)
    y_true =labels.attack
    y_pred=labels.pred
    
    
    # Adapt!!!! for pred and true
    TSNE_visualiser(data, labels, colour_by_level=2, marker_list=[], perplexity=30)

def build_and_evaluate_anomaly_detector(dataset, 
                                        training_samples=10000, 
                                        training_data_contamination=0.1, 
                                        IF_contamination=0.2,
                                        with_PCA = False):
    
    # Downsample training set according to contamination
    X_train = dataset["data_train"]
    y_train = dataset["labels_train"]
    
    # Define X_test, y_test
    X_test = dataset["data_test1"]
    y_test = dataset["labels_test1"]#["attack"]
    
    # PCA option
    if with_PCA:
        print("Using PCA to transform data")
        # Fit_transform training set
        scaler=StandardScaler()
        X_train_scaled=scaler.fit_transform(X_train)
        pca = PCA(n_components=None)
        X_train=pca.fit_transform(X_train_scaled)
        # Apply to test set
        X_test_scaled=scaler.transform(X_test)
        X_test=pca.transform(X_test_scaled)
        
    
    # Define IF-model
    clf_IF= IsolationForest(n_estimators=1000, 
                     max_samples='auto', 
                     max_features=float(1.0), 
                     contamination=float(IF_contamination), 
                     random_state=42, 
                     behaviour="new") 
    
    # Train and evaluate model
    kdd_train_evaluate_report_clf(X_train, X_test, y_test, clf_IF)
    #
    visualise_test_data(dataset)