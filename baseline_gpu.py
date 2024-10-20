import contextlib
import time

from concurrent.futures.thread import ThreadPoolExecutor as TPool

import jax
import mujoco as mj
from mujoco import mjx

from ant_xml import ANT_XML


@contextlib.contextmanager
def timer(name: str):
    t = .0
    try:
        t = time.perf_counter()
        print(f"{name} ...")
        yield None
    finally:
        print(f"'{name}' took {time.perf_counter() - t}")


def make_jit_step(timed=True):

    with timer("making jit step"):
        jit_step = jax.jit(mjx.step)

    if timed:
        with timer("making model"):
            model = mj.MjModel.from_xml_string(ANT_XML)
            data = mj.MjData(model)
            mjx_model = mjx.put_model(model)
            mjx_data = mjx.put_data(model, data)

        with timer("running first step"):
            jit_step(mjx_model, mjx_data)

        with timer("running second step"):
            jit_step(mjx_model, mjx_data)

        return jit_step


def run_scene_mjx(model: mj.MjModel, n_steps: int, jit_step):
    t = time.perf_counter()

    data = mj.MjData(model)
    mjx_model = mjx.put_model(model)
    mjx_data = mjx.put_data(model, data)

    for _ in range(n_steps):
        mjx_data = jit_step(mjx_model, mjx_data)

    return time.perf_counter() - t


def profile_single(scene_xml=ANT_XML):
    log = []
    attempts = 20
    model = mj.MjModel.from_xml_string(scene_xml)
    jit_step = make_jit_step()

    for n_steps in 1000, 2000, 4000:
        speeds = []
        for att in range(attempts):
            print(f"att: {att+1}/{attempts} on {n_steps=}...")
            speeds.append(run_scene_mjx(model, n_steps, jit_step))

        log.append(f'{n_steps=}, avg: {sum(speeds) / attempts}')

    print('results'.center(max(map(len, log)), '='))
    print('\n'.join(log))


def profile_n_threading(scene_xml: str, threads: int, scenes: int):
    log = []
    model = mj.MjModel.from_xml_string(scene_xml)
    jit_step = make_jit_step()

    for n_steps in 1000, 2000, 4000:
        with TPool(threads) as pool:
            futures = [pool.submit(run_scene_mjx, model=model, n_steps=n_steps, jit_step=jit_step)
                       for _ in range(scenes)]

            speeds = [future.result() for future in futures]
            log.append(f'{n_steps=}, avg: {sum(speeds) / len(speeds)}')

    print('results'.center(max(map(len, log)), '='))
    print('\n'.join(log))


if __name__ == '__main__':
    make_jit_step()
    # base_test()
    # profile_single()
    # profile_n_threading(ANT_XML, 100, 100)
