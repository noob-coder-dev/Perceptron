from utils.model import Perceptron
from utils.all_utils import prepare_data, save_model, save_plot
import pandas as pd
import numpy as np

def main(data, eta, epochs, modelFileName, plotFileName):
    OR = {
        'x1':[1,1,0,0],
        'x2':[1,0,1,0],
        'y':[1,1,1,0]
    }

    df = pd.DataFrame(data)

    print(df)

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

    main(data = OR, eta = eta, epochs = epochs, modelFileName = "or.model", plotFileName = "or.png")

