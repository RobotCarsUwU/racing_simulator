from mlagents_envs.base_env import ActionTuple, TerminalStep
import numpy as np
from pynput import keyboard
import csv
from datetime import datetime
import traceback
import pygame

up_pressed = False
down_pressed = False
left_pressed = False
right_pressed = False

def on_press(key):
    global up_pressed, down_pressed, left_pressed, right_pressed
    if key == keyboard.Key.up:
        up_pressed = True
    elif key == keyboard.Key.left:
        left_pressed = True
    elif key == keyboard.Key.right:
        right_pressed = True
    elif key == keyboard.Key.down:
        down_pressed = True

def on_release(key):
    global up_pressed, down_pressed, left_pressed, right_pressed
    if key == keyboard.Key.up:
        up_pressed = False
    elif key == keyboard.Key.left:
        left_pressed = False
    elif key == keyboard.Key.right:
        right_pressed = False
    elif key == keyboard.Key.down:
        down_pressed = False


def retrieveData(env):
    try:
        pygame.init()
        pygame.joystick.init()
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
                steering = joystick.get_axis(0)
                speed = joystick.get_axis(5)
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
