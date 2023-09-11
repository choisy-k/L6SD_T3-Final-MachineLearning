import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from joblib import load, dump
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

#Factory design pattern for models used
class ModelFactory:
    @staticmethod #responsible to create and return instances of different classess
    def get_model(choose):
        if choose == "Linear":
            return LinearRegression()
        elif choose == "Lasso":
            return Lasso()
        elif choose == "ElasticNet":
            return ElasticNet()
        elif choose == "Ridge":
            return Ridge()
        elif choose == "Support Vector":
            return SVR()
        elif choose == "Decision Tree":
            return DecisionTreeRegressor()        
        elif choose == "Random Forest":
            return RandomForestRegressor()
        elif choose == "K-Nearest Neighbors (KNN)":
            return KNeighborsRegressor()
        elif choose == "Gradient Boosting":
            return GradientBoostingRegressor()
        else:
            raise ValueError(f"{choose} Regression model is not found!")

model_names = [
    'Linear', 'Lasso', 'ElasticNet', 'Ridge', 'Support Vector', 'Decision Tree', 'Random Forest', 'K-Nearest Neighbors (KNN)', 'Gradient Boosting'
]

def load_data():
    print("============DATA DETECTION==============")
    df = pd.read_excel("Net_Worth_Data.xlsx")
    print("Loading data...")
    return df

def visualise_data(df):
    sns.set(style = "darkgrid")
    sns.pairplot(df)
    plt.tight_layout()
    plt.show()
    
def preprocess_data(df):
    #check for missing values
    print("Checking data for missing values...")
    if df.isnull().any().any():
        raise ValueError("Data cannot have missing values. Try change or use another data.")
    
    x_feature = df.drop(['Client Name', 'Client e-mail', 'Profession', 'Education', 'Country', 'Healthcare Cost', 'Net Worth'], axis=1)
    y_label = df['Net Worth']
    
    print("Transforming data...")
    x_sc = MinMaxScaler()
    y_sc = MinMaxScaler()
    
    x = x_sc.fit_transform(x_feature)
    y_reshape= y_label.values.reshape(-1,1)
    y = y_sc.fit_transform(y_reshape)
    return x, y

def split_data(x, y):
    print("Splitting data...")
    x_train, x_test, y_train, y_test = train_test_split(x, y.ravel(), test_size=0.2, random_state=42)
    print("Data successfully processed!")
    return x_train, x_test, y_train, y_test

def train_models(x_train, y_train):
    print("============LOADING MODELS==============")
    models = {}
    for name in model_names:
        print(f"Training {name} Regression model...")
        model = ModelFactory.get_model(name)
        model.fit(x_train, y_train)
        models[name] = model
        print("Success!")
    return models

def evaluate_models(x_test, y_test, models):
    rmse_values = {}
    
    for name, model in models.items():
        preds = model.predict(x_test)
        rmse_values[name] = mean_squared_error(y_test, preds, squared=False)
    return rmse_values

def plot_model_performance(rmse_values):
    models = list(rmse_values.keys())
    rmse = list(rmse_values.values())
    plt.figure(figsize=(10,7))
    bars = plt.barh(models, rmse, color=['green', 'cyan', 'magenta', 'yellow', 'orange', 'teal', 'brown', 'pink', 'gray'])

    # Add RMSE values on top of each bar
    for bar, val in zip(bars, rmse):
        plt.text(val, bar.get_y() + bar.get_height()/2, round(val, 5), ha='left', va='center', fontsize=10)

    plt.ylabel('Models')
    plt.xlabel('Root Mean Squared Error (RMSE)')
    plt.title('Model RMSE Comparison')
    plt.yticks(rotation=0, ha='right')
    plt.tight_layout()
    plt.show()

def save_best_model(models, rmse_values, x, y):
    # Getting the best model for the dataset
    best_model_name = min(rmse_values, key=rmse_values.get)
    print(f"Best model is {best_model_name} Regression.\nPlease wait while the model is retrained...")
    best_model = models[best_model_name]
    try:
        print("Retrain the model...")
        best_model.fit(x, y)
    except ValueError as ve:
        print(f"Error! {ve}")
        
    # Saving and loading best model
    print("Retraining completed. Proceeding to save trained model...")
    filename = "savedmodel_final.joblib"
    dump(best_model, filename)
    
    print(f"Model successfully saved as {filename}.")
    loaded_model = load(filename)
    return loaded_model

def predict_new_data(loaded_model):
    input_data = np.array([0, 42, 62812.09301, 11609.38091, 35321.45877, 75661.97242, 69670.45071, 42870.8743, 39218.97651, 58483.05364]).reshape(1, -1)
    predict_value = loaded_model.predict(input_data)
    print(f"Predicted Net Worth: ${predict_value[0][0]:.5f}")

if __name__ == "__main__":
    try:
        data = load_data()
        visualise_data(data)
        x, y = preprocess_data(data)
        x_train, x_test, y_train, y_test = split_data(x, y)
        models = train_models(x_train, y_train)
        rmse_values = evaluate_models(x_test, y_test, models)
        plot_model_performance(rmse_values)
        loaded_model = save_best_model(models, rmse_values, x, y)
        predict_new_data(loaded_model)
    except ValueError as ve:
        print(f"Error! {ve}")