from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from DataPreprocessing import Data

imagesPath = 'data/data/'
trainLabelsPath = 'data/train_labels.txt'
validationLabelsPath = 'data/validation_labels.txt'

if __name__ == '__main__':
    penalties = ['l2']
    C = [0.1, 1, 5, 10]
    solvers = ['lbfgs', 'newton-cg', 'saga']
    normalizations = ['standard', 'l1', 'l2']
    
    best_f1_score = 0.0
    
    data_loader = Data(imagesPath, trainLabelsPath, validationLabelsPath)
    train_data, train_labels, validation_data, validation_labels, test_data = data_loader.LoadData()
    
    class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=train_labels)
    class_weights = {0: class_weights[0], 1: class_weights[1]}
    
    best_model = None

    for normalization in normalizations:
        normalized_train_data, normalized_validation_data, normalized_test_data = data_loader.NormalizeData(train_data, validation_data,test_data, type=normalization)
        for solver in solvers:
            for c in C:
                for penalty in penalties:
                    model = LogisticRegression(penalty=penalty, solver=solver, C=c, class_weight=class_weights, n_jobs=-1)
                    model.fit(normalized_train_data, train_labels)
                    predicted_labels = model.predict(normalized_validation_data)
                    score = f1_score(validation_labels, predicted_labels)
                    if score > best_f1_score:
                        best_f1_score = score
                        best_model = model
                    print(f'Logistic Regression normalization={normalization}, solver={solver}, C={c}, penalty={penalty} got f1_score={score:.4f}')

    predicted_labels = best_model.predict(test_data)
    data_loader.OpenFile()
    data_loader.PrintData(predicted_labels)
    data_loader.CloseFile()