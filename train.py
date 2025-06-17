import numpy as np
import json
from Ai.NeuralNetwork.MLP.MLP import MLP
import tensorflow as tf
import keras


def load_data(csv_path):
    data = []
    labels = []
    with open(csv_path) as f:
        next(f)
        for line in f:
            values = [float(x) for x in line.strip().split(",")]
            labels.append(values[:2])
            data.append(values[2:])
    return np.array(data), np.array(labels)


def simple_normalize(X, stats_path="simple_stats.json", fit=True):
    """
    Normalisation simple min-max vers [0, 1]
    """
    if fit:
        min_vals = np.min(X, axis=0)
        max_vals = np.max(X, axis=0)

        range_vals = max_vals - min_vals
        range_vals = np.where(range_vals == 0, 1, range_vals)

        X_normalized = (X - min_vals) / range_vals

        stats = {
            "min": min_vals.tolist(),
            "max": max_vals.tolist(),
            "range": range_vals.tolist(),
        }
        with open(stats_path, "w") as f:
            json.dump(stats, f)
        print(f"Stat saved in {stats_path}")

        return X_normalized, stats
    else:
        with open(stats_path, "r") as f:
            stats = json.load(f)

        min_vals = np.array(stats["min"])
        range_vals = np.array(stats["range"])

        X_normalized = (X - min_vals) / range_vals

        return X_normalized, stats


def main():
    X, y = load_data("all_track_data_cleaned.csv")
    print(f"Data loaded : X shape={X.shape}, y shape={y.shape}")

    X_normalized, stats = simple_normalize(X, fit=True)

    model = MLP(input_size=X.shape[1], hidden_sizes=[128, 64, 32], output_size=2)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"],
    )

    model.build((None, X.shape[1]))
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=8, min_lr=1e-7
        ),
    ]

    history = model.fit(
        X_normalized,
        y,
        epochs=75,
        batch_size=64,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1,
    )

    print("Model trained !")
    model.save("racing_model.keras")

    predictions = model.predict(X_normalized[:5])
    print("Exemples de prédictions:")
    for i in range(5):
        print(
            f"Real: [{y[i][0]:.3f}, {y[i][1]:.3f}] -> Predict: [{predictions[i][0]:.3f}, {predictions[i][1]:.3f}]"
        )


if __name__ == "__main__":
    main()
