from utils.model import Perceptron
from utils.all_utils import prepare_data
import pandas as pd
import numpy as np

   AND = {
    'x1':[1,1,0,0],
    'x2':[1,0,1,0],
    'y':[1,0,0,0]
}

df = pd.DataFrame(AND)

X, y = prepare_data(df)

eta = 0.3 # 0 and 1
epochs = 10

model = Perceptron(eta=eta, epochs=epochs)
model.fit(X, y)

_ = model.total_loss()