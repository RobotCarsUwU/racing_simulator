# Script principal - compatible Python 3.6
import numpy as np
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Vérification de compatibilité
print(f"Python version requise: 3.6+")
print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")

def load_data(csv_path):
    """
    Charge les données depuis un fichier CSV
    """
    data = []
    labels = []
    with open(csv_path, 'r') as f:
        next(f)  # Skip header
        for line in f:
            values = [float(x) for x in line.strip().split(",")]
            labels.append(values[:2])
            data.append(values[2:])
    return np.array(data), np.array(labels)

def simple_normalize(X, stats_path="simple_stats.json", fit=True):
    """
    Normalisation simple min-max vers [0, 1]
    Compatible Python 3.6
    """
    if fit:
        min_vals = np.min(X, axis=0)
        max_vals = np.max(X, axis=0)
        range_vals = max_vals - min_vals
        # Éviter la division par zéro
        range_vals = np.where(range_vals == 0, 1, range_vals)
        X_normalized = (X - min_vals) / range_vals
        
        stats = {
            "min": min_vals.tolist(),
            "max": max_vals.tolist(),
            "range": range_vals.tolist(),
        }
        
        with open(stats_path, "w") as f:
            json.dump(stats, f)
        print("Statistiques sauvegardées dans {}".format(stats_path))
        return X_normalized, stats
    else:
        with open(stats_path, "r") as f:
            stats = json.load(f)
        min_vals = np.array(stats["min"])
        range_vals = np.array(stats["range"])
        X_normalized = (X - min_vals) / range_vals
        return X_normalized, stats

class MLP(keras.Model):
    """
    Multi-Layer Perceptron compatible Python 3.6 et TensorFlow 1.x/2.x
    """
    def __init__(self, input_size, hidden_sizes, output_size, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # Couches cachées
        self.hidden_layers = []
        for hidden_size in hidden_sizes:
            self.hidden_layers.append(layers.Dense(
                hidden_size,
                activation='relu',
                kernel_initializer='he_normal',
                kernel_regularizer=keras.regularizers.l2(1e-4)
            ))
            self.hidden_layers.append(layers.Dropout(0.2))
        
        # Couches de sortie
        self.speed_output = layers.Dense(1, activation='sigmoid', name='speed')
        self.steering_output = layers.Dense(1, activation='tanh', name='steering')
    
    def call(self, inputs, training=None):
        """
        Forward pass du modèle
        """
        x = inputs
        for layer in self.hidden_layers:
            if isinstance(layer, layers.Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)
        
        speed = self.speed_output(x)
        steering = self.steering_output(x)
        
        # Concatenation compatible TensorFlow 1.x/2.x
        return tf.concat([speed, steering], axis=1)
    
    def get_config(self):
        """
        Configuration pour la sérialisation
        """
        config = super(MLP, self).get_config()
        config.update({
            "input_size": self.input_size,
            "hidden_sizes": self.hidden_sizes,
            "output_size": self.output_size
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """
        Création depuis la configuration
        """
        return cls(**config)

def main():
    """
    Fonction principale d'entraînement
    """
    try:
        # Chargement des données
        X, y = load_data("all_track_data_cleaned_new.csv")
        print("Données chargées : X shape={}, y shape={}".format(X.shape, y.shape))
        
        # Normalisation
        X_normalized, stats = simple_normalize(X, fit=True)
        
        # Création du modèle
        model = MLP(input_size=X.shape[1], hidden_sizes=[128, 64, 32], output_size=2)
        
        # Compilation avec optimiseur compatible
        if hasattr(keras.optimizers, 'Adam'):
            optimizer = keras.optimizers.Adam(lr=0.001)  # lr au lieu de learning_rate pour TF 1.x
        else:
            optimizer = keras.optimizers.adam(lr=0.001)
        
        model.compile(
            optimizer=optimizer,
            loss="mse",
            metrics=["mae"],
        )
        
        # Construction du modèle
        model.build((None, X.shape[1]))
        
        # Affichage du résumé si disponible
        try:
            model.summary()
        except:
            print("Résumé du modèle non disponible")
        
        # Callbacks
        callbacks = []
        
        # EarlyStopping
        if hasattr(keras.callbacks, 'EarlyStopping'):
            callbacks.append(keras.callbacks.EarlyStopping(
                monitor="val_loss", 
                patience=15, 
                restore_best_weights=True
            ))
        
        # ReduceLROnPlateau
        if hasattr(keras.callbacks, 'ReduceLROnPlateau'):
            callbacks.append(keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", 
                factor=0.5, 
                patience=8, 
                min_lr=1e-7
            ))
        
        # Entraînement
        print("Début de l'entraînement...")
        history = model.fit(
            X_normalized,
            y,
            epochs=75,
            batch_size=64,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1,
        )
        
        print("Modèle entraîné !")
        
        # Sauvegarde
        try:
            model.save_weights("racing_model.h5")  # Format H5 plus compatible
            print("Modèle sauvegardé en racing_model.h5")
        except:
            print("Erreur lors de la sauvegarde")
        
        # Prédictions d'exemple
        predictions = model.predict(X_normalized[:5])
        print("Exemples de prédictions:")
        for i in range(5):
            print("Réel: [{:.3f}, {:.3f}] -> Prédiction: [{:.3f}, {:.3f}]".format(
                y[i][0], y[i][1], predictions[i][0], predictions[i][1]
            ))
            
    except Exception as e:
        print("Erreur durant l'exécution: {}".format(str(e)))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
