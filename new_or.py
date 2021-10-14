from perceptron.perceptron_class import Perceptron
#from utils.model import Perceptron
from utils.all_utils import prepare_data, save_model, save_plot
import pandas as pd
import numpy as np
import logging
import os

logging_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename = os.path.join(log_dir, "running_logs.log"), level=logging.INFO, format=logging_str, filemode="a")

def main(data, eta, epochs, modelFileName, plotFileName):
    

    df = pd.DataFrame(data)

    logging.info(f"This is the actual dataframe \n{df}")

    X, y = prepare_data(df)


    model = Perceptron(eta=eta, epochs=epochs)
    model.fit(X, y)

    _ = model.total_loss()

    save_model(model, filename="or.model")
    save_plot(df, filename="or.png", model=model)


if __name__ == "__main__":
    OR = {
        'x1':[1,1,0,0],
        'x2':[1,0,1,0],
        'y':[1,1,1,0]
    }

    eta = 0.3 # 0 and 1
    epochs = 10

    try:
        logging.info("\n>>>> TRAINING STARTED >>>>")
        main(data = OR, eta = eta, epochs = epochs, modelFileName = "or.model", plotFileName = "or.png")
        logging.info("<<<< TRAINING FINISHED <<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e
    

