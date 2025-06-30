import numpy as np
import json
import os
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
        print("Stats saved in {}".format(stats_path))
        return X_normalized, stats
    else:
        with open(stats_path, "r") as f:
            stats = json.load(f)
        min_vals = np.array(stats["min"])
        range_vals = np.array(stats["range"])
        X_normalized = (X - min_vals) / range_vals
        return X_normalized, stats

def create_or_load_model(input_size, pretrained_weights_path="racing_model.weights.h5", 
                        pretrained_stats_path="simple_stats.json"):
    model = MLP(input_size=input_size, hidden_sizes=[128, 64, 32], output_size=2)
    
    if os.path.exists(pretrained_weights_path):
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=0.0001),
            loss="mse",
            metrics=["mae"],
        )
        model.build((None, input_size))
        
        try:
            model.load_weights(pretrained_weights_path)
            return model, True
        except Exception as e:
            return model, False
    else:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=0.001),
            loss="mse",
            metrics=["mae"],
        )
        model.build((None, input_size))
        return model, False

def main():
    X, y = load_data("fine_tuning_data.csv")
    print("Data loaded: X shape={}, y shape={}".format(X.shape, y.shape))
    
    if os.path.exists("simple_stats.json"):
        X_normalized, stats = simple_normalize(X, fit=False)
    else:
        X_normalized, stats = simple_normalize(X, fit=True)
    
    model, is_fine_tuning = create_or_load_model(X.shape[1])
    
    if is_fine_tuning:
        epochs = 25
        patience = 8
        lr_patience = 5
        initial_lr = 0.0001
    else:
        epochs = 75
        patience = 15
        lr_patience = 8
        initial_lr = 0.001
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=initial_lr),
            loss="mse",
            metrics=["mae"],
        )
    
    model.summary()
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", 
            patience=patience, 
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", 
            factor=0.7,
            patience=lr_patience, 
            min_lr=1e-8,
            verbose=1
        ),
    ]
    
    callbacks.append(
        keras.callbacks.ModelCheckpoint(
            "racing_model_best.weights.h5",
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        )
    )
    
    history = model.fit(
        X_normalized,
        y,
        epochs=epochs,
        batch_size=32 if is_fine_tuning else 64,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1,
    )
    
    model.save_weights("racing_model.weights.h5")
    
    predictions = model.predict(X_normalized[:5])
    for i in range(5):
        print(
            "Real: [{:.3f}, {:.3f}] -> Predict: [{:.3f}, {:.3f}]".format(
                y[i][0], y[i][1], predictions[i][0], predictions[i][1]
            )
        )
    
    final_loss = history.history['val_loss'][-1]
    final_mae = history.history['val_mae'][-1]
    print(f"\n📈 Métriques finales:")
    print(f"   - Validation Loss: {final_loss:.6f}")
    print(f"   - Validation MAE: {final_mae:.6f}")

if __name__ == "__main__":
    main()
