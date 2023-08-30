import scipy.io
from scipy.io import loadmat
import numpy as np

# 1. Caricamento dei dati
def load_data(file_name):

    #if mat not already in extension,add it
    appendmat = file_name.endswith(".mat")
    mat_dict = loadmat(file_name,appendmat = appendmat)
    x = []
    y = []
    for raw in mat_dict["train"]:
        x_raw = raw[:-2]
        y_raw = raw[-2:-1]
        x.append(x_raw)
        y.append(y_raw)
    return x,y

# 2. Riduzione della dimensionalità
def dimensionality_reduction(X,Y):
    # Implementare il codice per la riduzione della dimensionalità
    pass

# 3. Classificatore logistico con gradiente stocastico
def logistic_classifier(X, Y, step_size):
    # Implementare il classificatore logistico
    pass

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
    train_file_name = "Gruppo 3/train.mat"
    test_file_name = "Gruppo 3/test.mat"
    x_train,y_train = load_data(file_name=train_file_name)
    x_test,y_test = load_data(file_name=train_file_name)
    #reduce dimensionality
    x_train_reduced = dimensionality_reduction(x_train,y_train)
    #fit classifier
    logistic_classifier = logistic_classifier(x_train_reduced,y_train)
    #predictions on test data
    predictions = make_predictions(x_test,logistic_classifier)
    #convert predictions to binary
    binary_bytes = decision_rule(predictions)
    #convert binary to ascii
    characters = to_ascii(binary_bytes)

    print(f"characters found :{characters}")

