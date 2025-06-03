from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple, TerminalStep
import numpy as np
import traceback

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
        env = createConnection()
        retrieveData(env)
        exit(0)
        # X, y = load_data('all_track_data_cleaned.csv')

        # model = MLP(input_size=50, hidden_size=[64, 32, 16], output_size=2, alpha=0.00005)
        # model.train(X, y, epochs=100, size=32)

        # env.reset()
        # behavior_name = list(env.behavior_specs.keys())[0]
        # decision_steps, _ = env.get_steps(behavior_name)

        # while True:
        #     decision_steps, _ = env.get_steps(behavior_name)
        #     if decision_steps.agent_id_to_index:
        #         raycast_data = decision_steps.obs[0]
        #         predicted = model.propagateForward(np.array(raycast_data))
        #         action = ActionTuple(continuous=np.array(predicted, dtype=np.float32))
        #         env.set_actions(behavior_name, action)
        #         env.step()

    except Exception as e:
        print("Error occurred:")
        traceback.print_exc()

    finally:
        env.close()

