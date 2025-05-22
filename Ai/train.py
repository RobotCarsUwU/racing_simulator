import numpy as np
from Ai.NeuralNetwork.MLP.MLP import MLP

def load_data(csv_path):
    data = []
    labels = []
    with open(csv_path) as f:
        next(f)  # Skip header if present
        for line in f:
            values = [float(x) for x in line.strip().split(',')]
            labels.append(values[:2])
            data.append(values[2:52])
    return np.array(data), np.array(labels)

def main():
    X, y = load_data('all_track_data.csv')
    model = MLP(input_size=50, hidden_size=[32, 16], output_size=2, alpha=0.001)
    model.train(X, y, epochs=100, size=32)
    # test_ = np.array([174.0,174.0,175.0,177.0,180.0,183.0,188.0,193.0,200.0,197.0,187.0,179.0,173.0,168.0,163.0,160.0,157.0,155.0,154.0,153.0,153.0,153.0,154.0,156.0,159.0,256.0,156.0,154.0,153.0,153.0,153.0,154.0,155.0,157.0,160.0,164.0,168.0,174.0,181.0,188.0,198.0,200.0,193.0,188.0,183.0,180.0,177.0,175.0,174.0,174.0])
    # test_2 = np.array([174.0,174.0,175.0,177.0,180.0,183.0,188.0,193.0,200.0,202.0,193.0,185.0,179.0,174.0,169.0,165.0,162.0,158.0,156.0,153.0,151.0,149.0,148.0,147.0,148.0,148.0,149.0,150.0,152.0,146.0,146.0,147.0,149.0,151.0,154.0,158.0,163.0,169.0,176.0,184.0,193.0,200.0,193.0,188.0,183.0,180.0,177.0,175.0,174.0,174.0])
    # prediction = model.propagateForward(test_)
    # print(prediction)
    # prediction = model.propagateForward(test_2)
    # print(prediction)
    

if __name__ == '__main__':
    exit(main())