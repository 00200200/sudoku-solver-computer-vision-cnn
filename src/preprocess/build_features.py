import src.common.tools as tools
import src.data.dataio as dataio
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split

def standardize_fit(X):
    # Standardize features by removing mean and scaling to unit variance:
    scaler = StandardScaler()
    return scaler.fit(X)

def standardize_transform(X, scaler):
    # Standardize features by removing mean and scaling to unit variance:
    return scaler.transform(X)

def split(X, y, test_fraction):
    # Split the data into a training and testing split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_fraction)
    return [X_train, X_test, y_train, y_test]

def preprocess(config):
    # Load the data
    rawdatapath = config["datarawdirectory"] + config["dataname"] + '.csv'
    [X, y] = dataio.load(rawdatapath)
    
    # Split the data
    test_fraction = 0.4
    [X_train, X_test, y_train, y_test] = split(X, y, test_fraction)
    
    # Save intermediate products
    savepath = config["datainterimdirectory"]
    dataio.save(X_train,y_train,savepath + "train.csv")
    dataio.save(X_test,y_test,savepath + "test.csv")
    
    # Standardize the data
    scaler = standardize_fit(X_train)
    X_train_scaled = standardize_transform(X_train, scaler)
    X_test_scaled = standardize_transform(X_test, scaler)
    
    # Save final products
    savepath = config["dataprocesseddirectory"]
    dataio.save(X_train_scaled,y_train,savepath + "train.csv")
    dataio.save(X_test_scaled,y_test,savepath + "test.csv")

if __name__ == "__main__":
    config = tools.load_config()
    preprocess(config)