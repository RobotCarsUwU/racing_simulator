from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple, TerminalStep
import numpy as np
import traceback
import tensorflow as tf
import keras

from Data.RetrieveData import retrieveData
from Ai.train import MLP, load_data

def createConnection():
    return UnityEnvironment(
        file_name="./RacingSimulatorLinux/RacingSimulator.x86_64",
        base_port=5004,
        seed=1,
        additional_args=['--config-path', './agent_config.json'])

def main():
    try:
        print("GPU:", tf.config.list_physical_devices('GPU'))
        
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"Erreur configuration GPU: {e}")

        X, y = load_data('all_track_data_cleaned.csv')
        print(f"Data loaded: X shape={X.shape}, y shape={y.shape}")

        model = MLP(
            hidden_sizes=[64, 32, 16], 
            output_size=2
        )
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.00005),
            loss='mse',
            metrics=['mae']
        )
        
        print("model:")
        model.summary()
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        history = model.fit(
            X, y, 
            epochs=100, 
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        print("Model trained")
        
        model.save('racing_model.keras')
        print("Model saved as 'racing_model.keras'")
        
        predictions = model.predict(X[:100])
        print(f"example: {predictions[:5]}")
        
        run_simulation = input("launch sim (y/n): ").lower() == 'y'
        
        env = createConnection()
        env.reset()
        if run_simulation:
            behavior_name = list(env.behavior_specs.keys())[0]
            
            print("Starting...")
            step_count = 0
            max_steps = 1000
            
            while step_count < max_steps:
                decision_steps, terminal_steps = env.get_steps(behavior_name)
                
                if len(decision_steps) > 0:
                    raycast_data = decision_steps.obs[0]
                    
                    predictions = model.predict(raycast_data)
                    
                    action = ActionTuple(continuous=predictions.astype(np.float32))
                    env.set_actions(behavior_name, action)
                    env.step()
                    
                    step_count += 1
                    
                    if step_count % 100 == 0:
                        print(f"Étape {step_count}, Action: {predictions[0]}")
                
                if len(terminal_steps) > 0:
                    print("Epoch done")
                    env.reset()
                    step_count = 0
            
            env.close()
            print("Sim over")

    except KeyboardInterrupt:
        print("\nCancel")
    except Exception as e:
        print("Error:")
        traceback.print_exc()
    finally:
        env.close()
