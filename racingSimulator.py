from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple, TerminalStep
import numpy as np
import traceback
import tensorflow as tf
import keras
import json

from Data.RetrieveData import retrieveData
from train import load_data

def createConnection():
    return UnityEnvironment(
        file_name="./RacingSimulatorLinux/RacingSimulator.x86_64",
        base_port=5004,
        seed=1,
        additional_args=['--config-path', './agent_config.json'])

def main():
    env = None
    try:
        print("GPU:", tf.config.list_physical_devices('GPU'))
        
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"Erreur configuration GPU: {e}")

        try:
            model = keras.models.load_model('racing_model.keras')
            print("Model loaded")
        except:
            print("Unable to load keras model")
            return
        
        try:
            with open('normalization_stats.json', 'r') as f:
                stats = json.load(f)
            mean = np.array(stats['mean'])
            std = np.array(stats['std'])
            print("Stat loaded")
        except:
            print("Unable to load stat")
            return
        
        env = createConnection()
        env.reset()
        
        behavior_name = list(env.behavior_specs.keys())[0]
        
        print("Starting simu...")
        step_count = 0
        max_steps = 2000
        
        while step_count < max_steps:
            decision_steps, terminal_steps = env.get_steps(behavior_name)
            
            if len(decision_steps) > 0:
                raycast_data = decision_steps.obs[0]
                
                raycast_normalized = (raycast_data - mean) / std
                
                predictions = model.predict(raycast_normalized, verbose=0)
                
                speed = np.clip(predictions[0][0], 0.0, 0.8)
                steering = np.clip(predictions[0][1], -0.8, 0.8)
                
                action = ActionTuple(continuous=np.array([[speed, steering]], dtype=np.float32))
                env.set_actions(behavior_name, action)
                env.step()
                
                step_count += 1
                
                if step_count % 50 == 0:
                    print(f"Étape {step_count}, Speed: {speed:.3f}, Steering: {steering:.3f}")
            
            if len(terminal_steps) > 0:
                print("Epoch ended")
                env.reset()
                step_count = 0

    except KeyboardInterrupt:
        print("\nSim interrupted")
    except Exception as e:
        print("Error during sim")
        traceback.print_exc()
    finally:
        if env:
            env.close()

if __name__ == "__main__":
    main()
