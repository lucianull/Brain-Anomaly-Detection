from sklearn import tree
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from DataPreprocessing import Data

imagesPath = 'data/data/'
trainLabelsPath = 'data/train_labels.txt'
validationLabelsPath = 'data/validation_labels.txt'

if __name__ == '__main__':
    data_loader = Data(imagesPath, trainLabelsPath, validationLabelsPath)
    train_data, train_labels, validation_data, validation_labels, test_data = data_loader.LoadData()    # incarcam datele
    train_data, validation_data, test_data = data_loader.NormalizeData(train_data, validation_data, test_data, type='standard')     # normalizam datele
    
    class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=train_labels)     # calculam weight-urile pentru fiecare clasa
    class_weights = {0: class_weights[0], 1: class_weights[1]}
    
    best_f1_score = 0.0
    best_model = None
    for i in range(5, 51, 5):
        model = tree.DecisionTreeClassifier(criterion='gini', class_weight=class_weights, max_depth=i)
        model.fit(train_data, train_labels)
        predicted_labels = model.predict(validation_data)
        score = f1_score(validation_labels, predicted_labels)      
        if score > best_f1_score:
            best_f1_score = score
            best_model = model
    
    predicted_labels = best_model.predict(test_data)        # printam predictiile pentru datele de testare ale celui mai bun model
    data_loader.OpenFile()
    data_loader.PrintData(predicted_labels)
    data_loader.CloseFile()