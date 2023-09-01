import scipy.io
from scipy.io import loadmat
import numpy as np
import os
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import random

#Caricamento dei dati
def load_data(file_name):
    #if mat not already in extension,add it
    appendmat = file_name.endswith(".mat")
    file = file_name.split(".")[0].split('/')[-1]
    mat_dict = loadmat(file_name,appendmat = appendmat)
    x = []
    y = []
    for raw in mat_dict[file]:
        if file == "train":
            x_raw = raw[:-1]
            y_raw = raw[-1:]
            x.append(x_raw)
            y.append(y_raw[0])
        else: 
            x.append(raw[:-1])
            y = None
    print('Features vectors', len(x))
    print('Features per array', len(x[0]))
    if y is not None:
        print('Labels', len(y))
        trues = 0
        falses = 0
        for i in y:
            if i == 1.0:
                trues = trues + 1
            elif i == -1.0:
                falses = falses + 1
        print('Trues', trues)
        print('Falses', falses)
    return x,y

def find_outliers(x):
    outliers = []
    m, _ = x.shape
    l = []
    for i in range(m):
        l.append(sum(x[i,:]))
    xt = np.sort(l)
    xt = list(xt)
    print("Rows tot after reduction:", len(xt))
    l1 = 0.25 * len(xt)
    l3 = 0.75 * len(xt)
    q1 = xt[round(l1)]
    q3 = xt[round(l3)]
    iqr = q3 - q1
    print('q1',q1,'q3',q3,'iqr',iqr)
    upb = q3 + 1.5 * iqr
    lowb = q1 - 1.5 * iqr
    for i in range(len(xt)):
        if xt[i] > upb or xt[i] < lowb:
            outliers.append(xt[i])
    print('Outliers numb. ', len(outliers))
    print('==============================================')
    return outliers

#Riduzione della dimensionalità
def correlation_dimensionality_reduction(x, min_corr=0.75):
    X = pd.DataFrame.from_dict(x)
    # Step 1: Compute the correlation matrix
    corr_matrix = X.corr().abs()
    # Step 2: Identify correlated variables
    upper = corr_matrix.where(pd.DataFrame(np.triu(np.ones(corr_matrix.shape), k=1)).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > min_corr)]
    # Step 3: Remove correlated variables
    X_reduced = X.drop(columns=to_drop)
    # Print information about deleted variables
    print(f"After correlation deleted {len(to_drop)} variables: {to_drop}")
    return X_reduced,to_drop

def drop_columns(x,columns_to_drop):
    X = pd.DataFrame.from_dict(x)
    X_reduced = X.drop(columns=columns_to_drop)
    return X_reduced

def pca_dimensionality_reduction(X,file_path,min_variance=0.90):
    features_standardized = standardization(X)
    n_components = len(features_standardized[0])
    pca_scaled = PCA(n_components)
    pca_scaled.fit(features_standardized)
    cumulative_variance_scaled = np.cumsum(pca_scaled.explained_variance_ratio_)
    plot_variance(pca_scaled.explained_variance_ratio_,cumulative_variance_scaled, np.arange(n_components), 'Principal components scaled', 'Variance ratio','variance on features standardized',file_path=file_path)
    # Find the number of components for at least min variance
    n_components_min_variance = np.argmax(cumulative_variance_scaled >= min_variance) + 1
    print(f"After PCA deleted {n_components-n_components_min_variance}")
    # Implementare il codice per la riduzione della dimensionalità
    pca_reducer = PCA(n_components_min_variance)
    features_reduced = pca_reducer.fit_transform(features_standardized)
    return features_reduced,pca_reducer
    
def standardization(features): 
    standard_scaler = preprocessing.StandardScaler()
    scaler_trained = standard_scaler.fit(features)
    features_standardized = scaler_trained.transform(features)
    return features_standardized

def plot_variance(variance, cum_variance, interval, x_label, y_label,title,file_path):  
    plt.bar(interval, variance, alpha = 0.5, align='center', label = 'Individual variance')
    plt.step(interval, cum_variance, where='mid', label = 'Cumulative variance')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='best')
    plt.savefig(file_path)
    plt.clf()

def compute_gradient_logisticLoss(y, X, beta):
    y = np.array(y)
    X = np.array(X)
    grad = (-y * X.T) / (1 + np.exp(y * np.dot(X.T, beta)))
    loss = np.mean(np.log(1 + np.exp(-y * np.dot(X, beta)))) # Logistic loss
    return grad, loss

def logistic_class_SGD_logisticLoss(X, y, loss_path, lr=0.01, n_epochs=10, file_path='gradient_vs_epoch.png'):
    X = np.insert(X, 0, 1, axis=1)
    m, n = X.shape
    weights = np.random.normal(size=n)
    epoch_avg_gradients = []  # Store average gradients for each epoch
    loss_history = []
    for epoch in tqdm(range(n_epochs)):
        losses = []
        epoch_gradients = []
        randomlist = random.sample(range(0, m), m)
        for i in range(m):
            random_idx = randomlist[i]
            xi = X[random_idx:random_idx+1]
            xi = np.reshape(xi, n)
            yi = y[random_idx:random_idx+1]
            gradient, loss = compute_gradient_logisticLoss(yi, xi, weights)
            weights -= lr * gradient
            epoch_gradients.append(np.linalg.norm(gradient))
            losses.append(np.linalg.norm(loss))
        epoch_avg_gradient = np.mean(epoch_gradients)
        epoch_avg_gradients.append(epoch_avg_gradient)
        loss_history.append(np.mean(losses))
    plot_gradients(epoch_avg_gradients, file_path)
    plot_losses(loss_history, loss_path + str(lr) + '.png')
    return weights

def plot_gradients(gradient_norms, file_path):
    plt.plot(range(len(gradient_norms)), gradient_norms, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Average Gradient Norm')
    plt.title('Gradient Norm vs Epoch')
    plt.grid(True)
    plt.savefig(file_path)
    plt.clf()

def plot_losses(losses, file_path):
    plt.plot(range(len(losses)), losses, marker='o', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.grid(True)
    plt.savefig(file_path)
    plt.clf()

def logistic_classifer(X, y, loss_path, step_size=0.1, epochs=50, file_path='logistic_classifier.png'):
    weights = logistic_class_SGD_logisticLoss(X, y, loss_path, step_size, epochs,file_path)
    X = np.insert(X, 0, 1, axis=1)
    predictions = logistic_inference(X, weights)
    return predictions, weights

def evaluate_step_sizes(X, y, loss_path, step_sizes, epochs, file_path):
    pred_best = None
    beta_best = None
    highest_accuracy = 0
    best_step_size = None
    accuracies = []
    for step_size in step_sizes:
        pred, beta = logistic_classifer(X, y, loss_path, step_size=step_size,epochs=epochs,file_path=os.path.join(images_path,'gradient_epochs_with_step_size_'+str(step_size)+'.png'))
        vett = [1 if pred[i]==y[i] else 0 for i in range(len(y))]
        accuracy = sum(vett)/len(vett)
        print('Accuracy for', step_size, ':', accuracy)
        accuracies.append(round(accuracy,5))
        # Store the best model
        if accuracy >= highest_accuracy:
            highest_accuracy = accuracy
            beta_best = beta
            pred_best = pred
            best_step_size = step_size
    print('==============================================')
    print(f"Best Logistic Classifier Accuracy with step size {best_step_size}: {highest_accuracy}")
    plot_accuracy_step_sizes(accuracies,step_sizes,file_path)
    return pred_best, beta_best

def plot_accuracy_step_sizes(accuracies,step_sizes,file_path):
    # Plotting
    plt.plot(range(len(step_sizes)), accuracies, marker='o')  # Change here
    plt.xlabel('Step Size')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Step Size')
    plt.grid(True)
    plt.xticks(range(len(step_sizes)), [str(s) for s in step_sizes])
    # Annotate each point
    for i, (x, y) in enumerate(zip(step_sizes, accuracies)):
        plt.annotate(str(y), (i, y))  # Use index i as x-coordinate
    plt.savefig(file_path)  # Save the plot to a file
    plt.clf()  # Clear the plot

# Inference function for new X values
def logistic_inference(X_test, best_beta):
    predictions = []
    for i, x_value in enumerate(X_test):
        res = np.dot(x_value, best_beta)
        if res > 0:
            predictions.append(1.0)
        else:
            predictions.append(-1.0)
    return predictions
    
# 6. Regola di decisione
def decision_rule(predictions):
    # Implementare la regola di decisione
    binaries = []
    for prediction in predictions:
        if prediction == -1.0:
            binaries.append(0)
        else:
            binaries.append(1)
    return binaries

# 7. Conversione in ASCII
def to_ascii(binary_bytes):
    binary_str = ''.join(map(str, binary_bytes))
    # Split the binary string into chunks of 8 bits and convert each chunk to ASCII
    ascii_chars = [chr(int(binary_str[i:i+8], 2)) for i in range(0, len(binary_str), 8)]
    return ascii_chars
    
def plot_correlation_matrix(x,filename):
    df = pd.DataFrame.from_dict(x)
    plt.figure("Correlation matrix")
    sns.heatmap(df.corr())
    plt.savefig(filename)
    plt.clf()

def remove_outliers(outs,x,y):
    print("Before outliers: X ", x.shape, "- y", len(y))
    new_y = []
    new_x = []
    m,n = x.shape
    for i in range(m):
        if sum(x[i,:]) not in outs:
            new_x.append(x[i,:])
            new_y.append(y[i])
    new_x = np.array(new_x)
    print("After outliers: X ", new_x.shape, "- y", len(new_y))
    print('==============================================')
    return new_x, new_y

if __name__ == "__main__":
    images_path = os.path.join(os.getcwd(),'Images')
    if os.path.exists(images_path) is False:
        os.makedirs(images_path)
    #load data
    print('==============================================')
    print("Train data:")
    train_file_name = "Gruppo 3/train.mat"
    x_train,y_train = load_data(file_name=train_file_name)
    print('==============================================')
    print("Test data:")
    test_file_name = "Gruppo 3/test.mat"
    x_test,y_test = load_data(file_name=test_file_name)
    plot_correlation_matrix(x_train,filename=os.path.join(images_path,'correlation_matrix_before_dim_reduction.png'))
    print('==============================================')
    print("Dimensionality Reduction...")
    print("Correlation...")
    #reduce dimensionality
    x_corr_reduced,columns_dropped= correlation_dimensionality_reduction(x_train)
    x_test_corr_reduced = drop_columns(x_test,columns_dropped)
    print("Features arrays reduced:", len(x_corr_reduced))
    print("Features per train array reduced:", x_corr_reduced.shape[1])
    plot_correlation_matrix(x_corr_reduced,filename=os.path.join(images_path,'correlation_matrix_after_corr_reduction.png'))
    print('==============================================')
    print("PCA...")
    x_corr_pca_reduced,pca_reducer = pca_dimensionality_reduction(x_corr_reduced,file_path=os.path.join(images_path,'dimensionality_reduction.png'))
    plot_correlation_matrix(x_corr_pca_reduced,filename=os.path.join(images_path,'correlation_matrix_after_corr_and_pca_reduction.png'))
    print("Features arrays reduced:", len(x_corr_pca_reduced))
    print("Features per train array reduced:", len(x_corr_pca_reduced[0]))
    print('==============================================')
    outs = find_outliers(x_corr_pca_reduced)
    x_corr_pca_reduced,y_train = remove_outliers(outs,x_corr_pca_reduced,y_train)
    print('Test')
    #step sizes
    step_sizes = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
    #find best logistic classifier
    best_bias,best_beta = evaluate_step_sizes(x_corr_pca_reduced,y_train,os.path.join(images_path,'Loss for stepsize '),step_sizes,epochs = 20,file_path=os.path.join(images_path,'accuracy on step size.png'))
    best_beta = best_beta[1:]
    x_test_corr_pca_reduced = pca_reducer.transform(x_test_corr_reduced)
    print('==============================================')
    print("Lenght x test reduced:",len(x_test_corr_pca_reduced[0]))
    #predictions on test data
    predictions = logistic_inference(x_test_corr_pca_reduced,best_beta)
    #print(predictions)
    #convert predictions to binary
    binary_bytes = decision_rule(predictions)
    #convert binary to ascii
    characters = to_ascii(binary_bytes)
    print('==============================================')
    print(f"characters found :{characters}")

