from mlagents_envs.base_env import ActionTuple, TerminalStep
import numpy as np
from pynput import keyboard

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
        keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        keyboard_listener.start()
        env.reset()
        behavior_name = list(env.behavior_specs.keys())[0]

        while True:
            steering = 0.0
            if left_pressed:
                steering = -0.5
            if right_pressed:
                steering = 0.5
            
            speed = 0.0
            if up_pressed:
                speed = 1.0
            if down_pressed:
                speed = -1.0

            action = ActionTuple(continuous=np.array([[speed, steering]], dtype=np.float32))

            decision_steps, Terminal_steps = env.get_steps(behavior_name)
            if decision_steps.agent_id_to_index:
                env.set_actions(behavior_name, action)
                env.step()

    except Exception:
        print("Error\n")

    finally:
        env.close()
