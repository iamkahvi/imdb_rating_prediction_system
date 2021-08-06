from sklearn.model_selection import  KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from keras.regularizers import l2, l1, l1_l2
import matplotlib.pyplot as plt
import keras
import pandas as pd
import numpy as np

np.random.seed(1337)


def simpleNNRegression(X, y): # number of hidden neurons selection analysis

    num_input = X.shape[1] # 956 dimensions
    num_samples = X.shape[0] # 22623 smaples
    num_output = 1 # 1 output neuron due to regression
    num_hidden_layers = 1 # 1 hidden layer in this NN


    # Researched methods for optimizing the number of neurons in the hidden layer
    num_hidden_neuron_methods = {
        'ShibataIkeda': int(np.sqrt(num_input * num_output)),
        'Trenn': int(num_input + num_output - 0.5),
        'JinchuanXinzhe': int(num_input + np.sqrt(num_samples) / num_hidden_layers)
    }

    results = []

    for method in num_hidden_neuron_methods:

        plt.figure(figsize=(12,12))
        plt.suptitle("5 Fold Cross Validation - Method for Number of Hidden Neurons to Use: " + method)
        subplot_idx = 1
        kf = KFold(n_splits=2, random_state=True, shuffle=True)
        kfold_results = {}
        for training_idx, testing_idx in kf.split(X):
            
            X_train, X_test = X.iloc[training_idx], X.iloc[testing_idx]
            y_train, y_test = y.iloc[training_idx], y.iloc[testing_idx]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            model = keras.Sequential([
                keras.layers.Dense(num_hidden_neuron_methods[method], input_dim=num_input, activation='sigmoid', kernel_initializer='he_normal'), #hidden
                keras.layers.Dense(num_output, activation="linear") # regression output layer
            ])

            model.compile(loss="mae", optimizer="adam", metrics=["mse"])
            print('Model Summary - CV Split ' + str(subplot_idx))
            print(model.summary())
            losses = model.fit(X_train, y_train, epochs=10, batch_size=num_input, validation_data=(X_test, y_test), verbose=10).history
            
        
            mae = min(losses['val_loss'])
            kfold_results['cv_split_' + str(subplot_idx)] = {'val_mae': mae, 'num_hidden': num_hidden_neuron_methods[method], 'method': method}

            plt.subplot(2,3,subplot_idx)
            plt.plot(losses['loss'])
            plt.plot(losses['val_loss'])
            plt.legend(['Training', 'Testing'], loc='upper right')
            plt.title('MAE Loss over 10 Epochs - CV Split ' + str(subplot_idx))
            plt.ylabel('MAE Loss')
            plt.xlabel('Epoch')
            subplot_idx += 1

        plt.draw()
        
        method_result = dict(sorted(kfold_results.items(), key=lambda result: result[1]['val_mae'], reverse=False))
        method_result = list(method_result.values())[0]
        results.append(method_result)
       

    plt.show()
    
    for result in results:

        plt.scatter(x=result['num_hidden'], y=result['val_mae'])
        plt.annotate(result['method'], (result['num_hidden'], result['val_mae']))

    plt.title('MAE For Each Number of Hidden Neuron Methods')
    plt.ylabel('MAE')
    plt.xlabel('Number of Hidden Neurons')
    plt.show()

    best_method = min(results, key=lambda result: result['val_mae'])
    print("The most effective method for choosing the number of hidden layer neurons:")
    print("Method: "+ best_method['method'])
    print("Number of Hidden Neurons: "+ str(best_method['num_hidden']))
    print("Validation MAE: "+ str(best_method['val_mae']))
    print()

    


def simpleNNClassification(X, y):

    num_input = X.shape[1] # 956 dimensions
    num_output = 100 # 100 output neurons for decimal tenths for ratings from 0.0-10.0
    num_hidden = int(num_input + num_output - 0.5) # setting optimal number of hidden layer neurons from simpleNNRegression


    # Variety of hyperparameters to test different combinations of
    activation_hyperparameter = [
        'sigmoid',
        'relu'
    ]
    kernel_initializer_hyperparameters = [
        'normal',
        'he_normal'
    ]


    results = []

    for activation in activation_hyperparameter:
        for kernel_initializer in kernel_initializer_hyperparameters:

            plt.figure(figsize=(12,12))
            plt.suptitle("5 Fold Cross Validation - Method for Hidden Activation: " + activation + " and Kernel Initializer: " + kernel_initializer)
            subplot_idx = 1
            kf = KFold(n_splits=2, random_state=True, shuffle=True)
            kfold_results = {}
            for training_idx, testing_idx in kf.split(X):
                
                X_train, X_test = X.iloc[training_idx], X.iloc[testing_idx]
                y_train, y_test = y.iloc[training_idx], y.iloc[testing_idx]

                # Encode rating range 0.0-10.0 over 100 classes
                y_train = to_categorical(y_train, 100)
                y_test = to_categorical(y_test, 100)

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                model = keras.Sequential([
                    keras.layers.Dense(num_hidden, input_dim=num_input, activation=activation, kernel_initializer=kernel_initializer), #hidden
                    keras.layers.Dense(num_output, activation="softmax") # 100 classes output layer
                ])

                model.compile(loss=keras.losses.BinaryCrossentropy(), metrics=["accuracy"])

                accuracies = model.fit(X_train, y_train, epochs=10, batch_size=num_input, validation_data=(X_test, y_test), verbose=10).history
                
            
                accuracy = max(accuracies['val_accuracy'])
                kfold_results['cv_split_' + str(subplot_idx)] = {'accuracy': accuracy, 'activation': activation, 'kernel_initializer': kernel_initializer}

                plt.subplot(2,3,subplot_idx)
                plt.plot(accuracies['accuracy'])
                plt.plot(accuracies['val_accuracy'])
                plt.legend(['Training', 'Testing'], loc='upper right')
                plt.title('Accuracy over 10 Epochs - CV Split ' + str(subplot_idx))
                plt.ylabel('Accuracy')
                plt.xlabel('Epoch')
                
                subplot_idx += 1
                

            plt.draw()
            
            method_result = dict(sorted(kfold_results.items(), key=lambda result: result[1]['accuracy'], reverse=True))
            method_result = list(method_result.values())[0]
            results.append(method_result)
            

    plt.show()
    i = 1
    for result in results:

        plt.scatter(x=i, y=result['accuracy'])
        annotation = "Activation: " + result['activation']+ " Kernel: " + result['kernel_initializer']
        plt.annotate(annotation, (i, result['accuracy']))
        i+=1

    plt.title('Accuracy For Each Hyperparameter Combination')
    plt.ylabel('Accuracy')
    plt.show()

    best_method = max(results, key=lambda result: result['accuracy'])
    print("The most accurate set of hyperparameters is:")
    print("Activation: "+ best_method['activation'])
    print("Kernel Initializer: "+ best_method['kernel_initializer'])
    print("Validation Accuracy: "+ str(best_method['accuracy']))
    print()

    
def classificationOverfittingAnalysis(X, y):

    num_input = X.shape[1] # 956 dimensions
    num_output = 100 # 100 output neurons for decimal tenths for ratings from 0.0-10.0
    num_hidden = int(num_input + num_output - 0.5)

    # setting optimal hyperparameters from simpleNNClassification
    activation = 'sigmoid' 
    kernel_initializer = 'normal'
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7)

    # Encode rating range 0.0-10.0 over 100 classes
    y_train = to_categorical(y_train, 100)
    y_test = to_categorical(y_test, 100)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    l2_regularizers = [0.0001, 0.000001]
    dropouts = [0.7, 0.3, 0.1]

    plt.figure(figsize=(16,16))
    plt.suptitle("Overfitting Analysis with Varying L2 and Dropout Hyperparameters")

    subplot_idx = 1
    for dropout_value in dropouts:
        for l2_value in l2_regularizers:
            
            
            model = keras.Sequential([
                keras.layers.Dense(num_hidden, input_dim=num_input, activation=activation, kernel_initializer=kernel_initializer, kernel_regularizer=l2(l2_value)),
                keras.layers.Dropout(dropout_value),
                keras.layers.Dense(num_output, activation="softmax") # 100 classes output layer
            ])
            
            model.compile(loss=keras.losses.BinaryCrossentropy(), metrics=["accuracy"])

            accuracies = model.fit(X_train, y_train, epochs=10, batch_size=num_input, validation_data=(X_test, y_test), verbose=10).history

            plt.subplot(2,3,subplot_idx)
            plt.plot(accuracies['accuracy'])
            plt.plot(accuracies['val_accuracy'])
            plt.legend(['Training', 'Testing'], loc='lower right')
            plt.title('Dropout: ' + str(dropout_value) + ' L2 Regulization: ' + str(l2_value))
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            
            subplot_idx += 1
            
    plt.show()



if __name__ == "__main__":

    df = pd.read_csv("content/combined_data.csv")

    y = df["weighted_average_vote"].astype(float)
    X = df.drop(
        ["imdb_title_id", "title", "weighted_average_vote"],
        axis=1,
    ).astype(float)


    # simpleNNRegression(X, y)
    # simpleNNClassification(X, y)
    classificationOverfittingAnalysis(X, y)
