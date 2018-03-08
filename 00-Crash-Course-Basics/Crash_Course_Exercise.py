import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def answer_one():
    np.random.seed(101)

def answer_two():
    answer_one()
    data = np.random.randint(1,101,(100,5))
    return data

def answer_three():
    data = answer_two()
    plt.imshow(data,aspect=0.05)
    plt.colorbar()
    plt.title("Data")
    plt.show()

def answer_four():
    data = answer_two()
    df = pd.DataFrame(data)
    return df

def answer_five():
    data = answer_two()
    plt.scatter(data[:,0], data[:,1])
    plt.show()

def answer_six():
    data = answer_two()
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

def answer_seven():
    df = answer_four()
    df.columns = ['f1','f2','f3','f4','label']
    X = df[['f1','f2','f3','f4']]
    y =  df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":

    answer_one()
    data = answer_two()
    answer_three()
    df = answer_four()
    answer_five()
    scaled_data = answer_six()
    X_train, X_test, y_train, y_test = answer_seven()