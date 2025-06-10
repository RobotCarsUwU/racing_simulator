import numpy as np
from Ai.NeuralNetwork.MLP.MLP import MLP

def load_data(csv_path):
    data = []
    labels = []
    with open(csv_path) as f:
        next(f)
        for line in f:
            values = [float(x) for x in line.strip().split(',')]
            labels.append(values[:2])
            data.append(values[2:52])
    return np.array(data), np.array(labels)

# def main():
#     X, y = load_data('all_track_data.csv')
#     model = MLP(input_size=50, hidden_size=[32, 16], output_size=2, alpha=0.001)
#     model.train(X, y, epochs=100, size=32)
