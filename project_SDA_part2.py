import scipy.io
from scipy.io import loadmat
import numpy as np
import os
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm


#Caricamento dei dati
def load_data(file_name):

    #if mat not already in extension,add it
    
    appendmat = file_name.endswith(".mat")
    file = file_name.split(".")[0].split(os.sep)[-1]
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
            x.append(raw[1:])
            y = None
            
    return x,y


#Riduzione della dimensionalità
def dimensionality_reduction(X,min_variance=0.9):

    features_standardized = standardization(X)
    n_components = len(features_standardized[0])
    pca_scaled = PCA(n_components)
    pca_scaled.fit(features_standardized)
    cumulative_variance_scaled = np.cumsum(pca_scaled.explained_variance_ratio_)
    plot_variance(pca_scaled.explained_variance_ratio_,cumulative_variance_scaled, np.arange(n_components), 'Principal components scaled', 'Variance ratio','variance on features standardized')

    # Find the number of components for at least min variance
    n_components_min_variance = np.argmax(cumulative_variance_scaled >= min_variance) + 1
    # Implementare il codice per la riduzione della dimensionalità
    pca_reducer = PCA(n_components_min_variance)
    features_reduced = pca_reducer.fit_transform(features_standardized)

    return features_reduced,pca_reducer
    

def standardization(features):
        
    standard_scaler = preprocessing.StandardScaler()
    scaler_trained = standard_scaler.fit(features)
    features_standardized = scaler_trained.transform(features)

    return features_standardized


def plot_variance(variance, cum_variance, interval, x_label, y_label,title):
    plt.bar(interval, variance, alpha = 0.5, align='center', label = 'Individual variance')
    plt.step(interval, cum_variance, where='mid', label = 'Cumulative variance')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='best')
    plt.show()


# Sigmoid function
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

# Cost function
def cost_function(y, h):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()


# Gradient Descent
def gradient_descent(X, y, theta, alpha, epochs):
    m = len(y)
    for _ in tqdm(range(epochs)):
        z = np.dot(X, theta)
        h = sigmoid(z)
        gradient = np.dot(X.T, np.subtract(h,y)) / m
        theta -= alpha * gradient
    return theta


#Prediction function
def predict(X, theta):
    predictions = []
    for x_value in X:
        if sigmoid(np.dot(x_value, theta)) >= 0.5:
            predictions.append(1.0)
        else:
            predictions.append(-1.0)
    return predictions


def logistic_classifer(X, y, step_size=0.1,epochs=10):

    # Add a bias term to the feature matrix
    X_bias = np.c_[np.ones(X.shape[0]), X]
   
    # Initialize theta for each step size
    theta = np.zeros(X_bias.shape[1])

    # Train the model with the current step size
    theta = gradient_descent(X_bias, y, theta, step_size, epochs)
    
    return X_bias,theta


def evaluate_step_sizes(X,y,step_sizes):

    x_bias_best = None
    theta_best = None
    highest_accuracy = 0
    best_step_size = None

    for step_size in step_sizes:
        X_bias,theta = logistic_classifer(X,y,step_size=step_size)
        y_pred = predict(X_bias, theta)
        vett = [1 if y_pred[i]==y[i] else 0 for i in range(len(y))]

        accuracy = sum(vett)/len(vett)
        print(accuracy)
        # Store the best model
        if accuracy > highest_accuracy:
            highest_accuracy = accuracy
            best_theta = theta
            best_X_bias = X_bias
            best_step_size = step_size

        print(f"Logistic Classifier Accuracy with step size {step_size}: {accuracy}")
   
    print(f"Best Logistic Classifier Accuracy with step size {best_step_size}: {accuracy}")
    return best_X_bias, best_theta

# Inference function for new X values
def logistic_inference(X_test, best_theta):
    # Add a bias term to the new feature matrix
    #new_X_bias = np.c_[np.ones(len(new_X)), new_X]
    
    # Predictions using the best theta
    #new_y_pred = predict(new_X_bias, best_theta)

    predictions = []
    for i,x_value in enumerate(X_test):
        res = x_value*best_theta
        print("=============")
        print(x_value,best_theta)
        if res > 0:
            predictions.append(1)
        else:
            predictions.append(-1)

    print(len(predictions))

    return predictions
    
    

# 6. Regola di decisione
def decision_rule(predictions):
    # Implementare la regola di decisione
    binaries = []
    for prediction in predictions:
        if prediction == -1:
            binaries.append(0)
        else:
            binaries.append(1)
    return binaries

# 7. Conversione in ASCII
def to_ascii(binary_bytes):
    print(binary_bytes)
    pass

if __name__ == "__main__":
    
    #load data
    print("Train data\n")
    train_file_name = "Gruppo 3/train.mat"
    x_train,y_train = load_data(file_name=train_file_name)
    print("Test data\n")
    test_file_name = "Gruppo 3/test.mat"
    x_test,y_test = load_data(file_name=test_file_name)
    #reduce dimensionality
    x_train_reduced,pca_reducer= dimensionality_reduction(x_train)
    print("len x train reduced:",len(x_train_reduced[0]))
    #step sizes
    step_sizes = [0.00001,0.0001,0.001, 0.01, 0.1, 1]
    #find best logistic classifier
    best_bias,best_beta = evaluate_step_sizes(x_train_reduced,y_train,step_sizes)
    
    best_beta = best_beta[1:]

    print(f"best beta:{best_beta}")
    x_test_reduced = pca_reducer.transform(x_test)
    print("len x test reduced:",len(x_test_reduced[0]))
    #predictions on test data
    predictions = logistic_inference(x_test_reduced,best_beta)
    print(predictions)
    #convert predictions to binary
    binary_bytes = decision_rule(predictions)
    #convert binary to ascii
    characters = to_ascii(binary_bytes)

    print(f"characters found :{characters}")

