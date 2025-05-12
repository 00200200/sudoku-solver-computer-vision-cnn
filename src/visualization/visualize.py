import src.common.tools as tools
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def confusion(y_true,y_pred,labels):
    # Create a confustion matrix from the results
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_true,y_pred,
                                            labels=labels,
                                            ax=ax,
                                            normalize='true',
                                            cmap='gray')
    return fig


def visualize_results(Results,outfolder):
    # Create and save all visualizations
    fig = confusion(Results.y_true,Results.y_pred,Results.classes)
    fig.savefig(outfolder + 'confusion.png', dpi=600)

if __name__ == "__main__":
    # Load the results
    config = tools.load_config()
    resultspath = config["resultsevaluatedpath"]
    Results = tools.pickle_load(resultspath)
    
    # Produce and save the figure
    figuredirectory = config["figuredirectory"]
    visualize_results(Results, figuredirectory)
    