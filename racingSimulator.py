from mlagents_envs.environment import UnityEnvironment
from Data.RetrieveData import retrieveData

def createConnection():
    return UnityEnvironment(
        file_name="./RacingSimulatorLinux/RacingSimulator.x86_64",
        base_port=5004,
        seed=1,
        additional_args=['--config-path', './agent_config.json'])

def main():
    env = createConnection()
    retrieveData(env)
