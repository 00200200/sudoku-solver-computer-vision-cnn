import src.common.tools as tools
from urllib.request import urlretrieve

def download_data(link,savepath):
    # Download the data from the web
    urlretrieve(link,savepath)

if __name__ == "__main__":
    # Download and save the raw data
    config = tools.load_config()
    savename = config["datarawdirectory"] + config["dataname"] + ".csv"
    download_data(config["datalink"],savename)