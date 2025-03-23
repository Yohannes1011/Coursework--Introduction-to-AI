'I have decided to structure the code for both of these projects as functions for each of the steps to make it easier for me to work through and understand. I also coded it in a  way where batch and stochastic gradient desent is done in one function and can be called at the end'


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

'''Loads the dataset in'''
def load():
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['MedianHouseValue'] = data.target
    return df

'''plots a scatter graph of MedInc and MedVal, It also Calculates the summmary statistics and leaves out the ones not needed.'''
def visualize(df):
    plt.scatter(df['MedInc'], df['MedianHouseValue'], color='green')
    plt.xlabel('Median Income')
    plt.ylabel('Median House Value')
    plt.title('Median Income vs Median House Value')
    plt.show()

    income = df[['MedInc']].agg(['mean', 'median', 'std'])
    houseval = df[['MedianHouseValue']].agg(['mean', 'median', 'std'])
    print("Median Income Stats:", income)
    print("Median House Value Stats:", houseval)

'''splits the data into training and test data at an 80/20 split'''
def split(df):
    X = df[['MedInc']].values
    y = df['MedianHouseValue'].values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)

    X_train = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    X_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]

    return X_train, X_test, y_train, y_test

'''calculates the Mean Squared Error for both which will be used in the gradient descents'''
def Cost1(theta, X, y):
    m = len(y)  
    predictions = X.dot(theta)  
    cost = (1 / (2 * m)) * np.sum(np.square(predictions - y))
    return cost

'''calculates the gradient'''
def gradient1(X, y, theta):
    m = len(y)
    return (1 / m) * X.T.dot(X.dot(theta) - y)

'''updates theta'''
def updatetheta(theta, gradient, learningrate):
    return theta - learningrate * gradient

'''Function that runs both batch and gradient descent'''
def gradientdescent(X, y, initialtheta, learningrate=0.01, iters=1000, mode='batch', tolerance=1e-5):
    theta = initialtheta.copy()
    m = len(y)
    costhistory = []

    np.random.seed(12)

    for i in range(iters):
        if mode == 'batch':
            gradient = gradient1(X, y, theta)
        elif mode == 'sgd':
            rand_index = np.random.randint(0, m)
            X_sample = X[rand_index, :].reshape(1, -1)
            y_sample = y[rand_index, :].reshape(1, -1)
            gradient = X_sample.T.dot(X_sample.dot(theta) - y_sample)
        else:
            print("Invalid mode selected.")
            break

        theta = updatetheta(theta, gradient, learningrate)
        cost = Cost1(theta, X, y)
        costhistory.append(cost)

        '''This checks for convergence'''
        if mode == 'batch' and len(costhistory) > 1 and abs(costhistory[-1] - costhistory[-2]) < tolerance:
            break

    return theta, costhistory

'''Makes predictions For the house values with the test set'''
def predict(theta, new_data):
    return new_data.dot(theta)

'''Plots those results with a regression line over the test data'''
def plot(X_test, y_test, theta):
    plt.scatter(X_test[:, 1], y_test, color='green', label='Test Data')
    y_pred = X_test.dot(theta)
    plt.plot(X_test[:, 1], y_pred, color='red', label='Regression Line')
    plt.xlabel('Median Income')
    plt.ylabel('Median House Value')
    plt.title('Linear Regression')
    plt.legend()
    plt.show()

'''This Final Function Runs all the previous ones in order and so it makes the code easier to run as we can simply call it at the end to run the whole project'''
def runitall(gradient_type='batch'):
    df = load()
    visualize(df)

    X_train, X_test, y_train, y_test = split(df)
    initial_theta = np.zeros((X_train.shape[1], 1))

    '''This allows me to change whether I run Stochastic or batch gradient descent'''
    if gradient_type == 'batch':
        theta, _ = gradientdescent(X_train, y_train, initial_theta, learningrate=0.0175, iters=850, mode='batch')
    elif gradient_type == 'sgd':
        theta, _ = gradientdescent(X_train, y_train, initial_theta, learningrate=0.0125, iters=750, mode='sgd')
    else:
        print("Invalid method. Choose 'batch' or 'sgd'.")
        return

    predictions = X_test.dot(theta)

    '''Makes the prediction with a median income of 80,000'''
    new = np.array([[1, 8.0]])  
    predictedvalue = new.dot(theta)

    print("Predictions for first 5 test samples:", predictions[:5])
    print("Predicted house value for district with median income $80,000:", predictedvalue[0])

    plot(X_test, y_test, theta)

'''At the end here I have written the code in a way that you can simply change from batch or sgd to run the type of gradient descent on the data and look at each to compare results.'''
runitall(gradient_type='batch')
