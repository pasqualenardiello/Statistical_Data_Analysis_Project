import scipy.io
from scipy.io import loadmat
import numpy as np
import os
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1. Caricamento dei dati
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
            y.append(y_raw)
        else: 
            x.append(raw)
            y = None
            
    return x,y

# 2. Riduzione della dimensionalità
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
    pca_reduced = PCA(n_components_min_variance)
    features_reduced = pca_reduced.fit_transform(features_standardized)

    return features_standardized
    

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


# 4. Valutazione dello step-size
def evaluate_step_size():
    # Implementare il codice per valutare diverse scelte dello step-size
    pass

# 5. Predizione
def make_predictions(test_data, model):
    # Implementare il codice per fare predizioni
    pass

# 6. Regola di decisione
def decision_rule(predictions):
    # Implementare la regola di decisione
    pass

# 7. Conversione in ASCII
def to_ascii(binary_bytes):
    # Implementare la conversione in ASCII
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
    x_train_reduced = dimensionality_reduction(x_train)
    #fit classifier
    logistic_classifier = logistic_classifier(x_train_reduced,y_train)
    #predictions on test data
    predictions = make_predictions(x_test,logistic_classifier)
    #convert predictions to binary
    binary_bytes = decision_rule(predictions)
    #convert binary to ascii
    characters = to_ascii(binary_bytes)

    print(f"characters found :{characters}")

