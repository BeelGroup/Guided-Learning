import random
import pyglet
import ctypes

import numpy as np

from pyglet import clock
from pyglet.window import key as keycodes
from pyglet.gl import *

from utils import *


def get_human_input(env):
    '''
    :param env: The Retro environment
    :return: A list of tuples: (obs, info, action)
    '''

    save_period = 10  # frames

    obs = env.reset()

    screen_height, screen_width = obs.shape[:2]

    random.seed(0)

    key_handler = pyglet.window.key.KeyStateHandler()
    win_width = 700
    win_height = win_width * screen_height // screen_width
    try:
        win = pyglet.window.Window(width=win_width, height=win_height, vsync=False)
    except:
        print("Exception Occurred: {}".format(sys.exc_info()[0]))
        return ([], [])

    if hasattr(win.context, '_nscontext'):
        pixel_scale = win.context._nscontext.view().backingScaleFactor()

    win.width = win.width // pixel_scale
    win.height = win.height // pixel_scale

    joysticks = pyglet.input.get_joysticks()
    if len(joysticks) > 0:
        joystick = joysticks[0]
        joystick.open()
    else:
        joystick = None

    win.push_handlers(key_handler)

    key_previous_states = {}

    pyglet.app.platform_event_loop.start()

    fps_display = pyglet.clock.ClockDisplay()
    clock.set_fps_limit(60)

    glEnable(GL_TEXTURE_2D)
    texture_id = GLuint(0)
    glGenTextures(1, ctypes.byref(texture_id))
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, screen_width, screen_height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)

    steps = 0
    recorded_io = []
    recorded_io_count = []

    started = False
    first_key = True
    timer_start = get_epochtime_ms()
    prev_action = None
    action_count = 1

    print("Waiting for input..")
    while True:
        win.dispatch_events()

        win.clear()

        keys_clicked = set()
        keys_pressed = set()
        key_clicked = False
        key_pressed = False
        for key_code, pressed in key_handler.items():
            if pressed:
                keys_pressed.add(key_code)
                key_pressed = True

            if not key_previous_states.get(key_code, False) and pressed:
                keys_clicked.add(key_code)
                key_clicked = True
            key_previous_states[key_code] = pressed

        if key_clicked and not started:
            print("Starting..")
            started = True

        if (keycodes.S in keys_clicked):
            print("Finished Recording")
            break

        if (keycodes.R in keys_clicked):
            print("Restarting..")
            obs = env.reset()
            steps = 0
            recorded_io = []
            recorded_io_count = []
            started = False
            first_key = True
            timer_start = get_epochtime_ms()
            prev_action = None
            action_count = 1
            continue

        if not started and get_epochtime_ms() - timer_start > 3 * 1000:
            # timeout
            break
        else:
            inputs = {
                'B': keycodes.X in keys_pressed,

            'None': False,

            'SELECT': keycodes.TAB in keys_pressed,
            'START': keycodes.ENTER in keys_pressed,

            'UP': keycodes.UP in keys_pressed,
            'DOWN': keycodes.DOWN in keys_pressed,
            'LEFT': keycodes.LEFT in keys_pressed,
            'RIGHT': keycodes.RIGHT in keys_pressed,

            'A': keycodes.Z in keys_pressed,
            }

            action = np.array([int(inputs[key]) for key in inputs]).astype(np.uint8)
            obs, rew, done, info = env.step(action)

            if key_clicked:
                if not first_key:
                    # record on every key change. Gives very few samples but this is fine because we want to overfit
                    recorded_io.append((obs, info, action))
                if first_key:
                    first_key = False

            if key_pressed:
                if prev_action != None:
                    if (action == prev_action).all():
                        action_count += 1
                    else:
                        recorded_io_count.append(action_count)
                        action_count = 1

                prev_action = action

            #if steps % save_period == 0 or key_clicked):

            steps += 1

        # Draw the game to the window
        glBindTexture(GL_TEXTURE_2D, texture_id)
        video_buffer = ctypes.cast(obs.tobytes(), ctypes.POINTER(ctypes.c_short))
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, obs.shape[1], obs.shape[0], GL_RGB, GL_UNSIGNED_BYTE, video_buffer)

        x = 0
        y = 0
        h = win.height
        w = win.width

        pyglet.graphics.draw(
            4,
            pyglet.gl.GL_QUADS,
            ('v2f', [x, y, x + w, y, x + w, y + h, x, y + h]),
            ('t2f', [0, 1, 1, 1, 1, 0, 0, 0]),
        )

        fps_display.draw()

        win.flip()

        # process joystick events
        timeout = clock.get_sleep_time(False)
        pyglet.app.platform_event_loop.step(timeout)

        clock.tick()

    pyglet.app.platform_event_loop.stop()
    win.close()
    pyglet.app.exit()

    recorded_io_count = recorded_io_count[1:] # remove the first entry (before the first action takes place)

    return recorded_io, recorded_io_count
