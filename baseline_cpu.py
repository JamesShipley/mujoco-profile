import time

import mujoco as mj
import numpy as np

from ant_xml import ANT_XML
from robot_xml import ROBOT_XML

CTRLS = np.sin(np.arange(10_000))


def run_scene(model: mj.MjModel, n_steps: int, control: bool):
    t = time.perf_counter()

    data = mj.MjData(model)
    mj.mj_forward(model, data)
    if control:
        print(data.ctrl.shape)
        data.ctrl = 1

    for i in range(n_steps):
        mj.mj_step(model, data)
        # if control:
        #     data.ctrl = 0
    # print(data.ctrl)
    return time.perf_counter() - t


def profile(scene_xml: str, control: bool):
    model = mj.MjModel.from_xml_string(scene_xml)
    attempts = 20

    log = []
    for n_steps in 1000, 2000, 4000:
        speeds = []
        for att in range(attempts):
            print(f"att: {att+1}/{attempts} on {n_steps=}...")
            speeds.append(run_scene(model, n_steps, control))
        avg = sum(speeds) / attempts
        log.append(f'{n_steps=}, avg: {avg}, per step: {avg/n_steps}')

    return log


def print_log(log):
    print('results'.center(max(map(len, log)), '='))
    print('\n'.join(log))


def mx():
    log1 = profile(ROBOT_XML, True)
    log2 = profile(ROBOT_XML, False)

    print_log(log1)
    print_log(log2)


if __name__ == '__main__':
    mx()
