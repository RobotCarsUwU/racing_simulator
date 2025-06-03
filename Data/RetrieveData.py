from mlagents_envs.base_env import ActionTuple, TerminalStep
import numpy as np
import csv
from datetime import datetime
import traceback
import pygame

def retrieveData(env):
    try:
        pygame.init()
        pygame.joystick.init()
        joystick_count = pygame.joystick.get_count()
        joystick = None
        if joystick_count > 0:
            joystick = pygame.joystick.Joystick(0)
            joystick.init()

        now = datetime.now()
        filename = f"data_{now.strftime('%Y-%m-%d')}_{now.strftime('%H-%M-%S')}.csv"

        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)

            env.reset()
            behavior_name = list(env.behavior_specs.keys())[0]

            decision_steps, _ = env.get_steps(behavior_name)
            raycast_headers = [f"Raycast {i}" for i in range(len(decision_steps.obs[0].flatten()))]
            writer.writerow(["Speed", "Steering"] + raycast_headers)

            while True:
                pygame.event.pump()
                if joystick:
                    steering = joystick.get_axis(0)
                    speed = joystick.get_axis(5)
                else:
                    keys = pygame.key.get_pressed()
                    speed = 0.0
                    steering = 0.0
                    if keys[pygame.K_z]:
                        speed = 1.0
                    elif keys[pygame.K_s]:
                        speed = 0.0
                    if keys[pygame.K_q]:
                        steering = -1.0
                    elif keys[pygame.K_d]:
                        steering = 1.0
                steering = max(-1.0, min(1, steering))
                speed = max(0.0, min(1.0, speed))
                if (-0.1 <= steering <= 0.1):
                    steering = 0

                action = ActionTuple(continuous=np.array([[speed, steering]], dtype=np.float32))

                decision_steps, _ = env.get_steps(behavior_name)
                if decision_steps.agent_id_to_index:
                    raycast_data = decision_steps.obs[0]
                    raycast_values = raycast_data.flatten().tolist()
                    writer.writerow([speed, steering] + raycast_values)
                    env.set_actions(behavior_name, action)
                    env.step()

    except Exception as e:
        print("Error occurred:")
        traceback.print_exc()

    finally:
        env.close()
        pygame.quit()
