import datetime
import multiprocessing
import time
from concurrent.futures.process import ProcessPoolExecutor as ProcPool

import mujoco
import numpy as np
import pandas as pd
from mujoco import mjx
import jax
import jax.numpy as jnp

from mjprofiler.bodies import Ant, Humanoid

POPULATION_SIZE = (100, 200, 400, 800, 1600, 3200)
N_STEPS = (100, 200, 400, 800, 1600, 3200)
BODIES = (Ant.make(10, 10, 10), Humanoid.make(10))
SIN = np.sin(np.arange(0, 2 * np.pi, 0.01))
N_PROCS = multiprocessing.cpu_count()


class Timer:
    start: float = -1
    elapsed: float = -1

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.perf_counter() - self.start


def _run(body_xml, n_steps, i_population):
    with Timer() as t:
        model = mujoco.MjModel.from_xml_string(body_xml)
        data = mujoco.MjData(model)

        for i_step in range(n_steps):
            mujoco.mj_step(model, data)
            data.ctrl = SIN[(i_population + i_step) % len(SIN)]
    return t.elapsed


def _cpu_sim_single(population: int, n_steps: int, body_xml: str, is_mp: bool) -> float:
    print(f"{population=} | {n_steps=}")

    with ProcPool(max_workers=N_PROCS) as pool:
        with Timer() as t:
            if is_mp:
                futures = [pool.submit(_run, body_xml, n_steps, i_population) for i_population in range(population)]
                compute_time = sum([fut.result() for fut in futures])
            else:
                compute_time = sum(_run(body_xml, n_steps, i_population) for i_population in range(population))

        total_time = t.elapsed
        print(f'total: {total_time}, compute: {compute_time}, util: {compute_time / total_time}')
        return compute_time


def main_cpu(body_xml: str, attempts: int):
    print(f"started {datetime.datetime.now()} with {N_PROCS} processes")

    x, y = len(POPULATION_SIZE), len(N_STEPS)
    results = np.zeros((x, y))

    for i_population in range(x):
        for i_steps in range(y):
            results[i_population, i_steps] = _cpu_sim_single(
                    POPULATION_SIZE[i_population],
                    N_STEPS[i_steps],
                    body_xml,
                    attempts,
                    is_mp=False
                )
    df = pd.DataFrame({f"{pop=}": results[i] for i, pop in enumerate(POPULATION_SIZE)}, index=list(N_STEPS))
    print(df.to_string())


def _gpu_sim_single(population: int, n_steps: int, body_xml: str):

    print(f"GPU | {population=} | {n_steps=}")
    model = mujoco.MjModel.from_xml_string(body_xml)
    data = mujoco.MjData(model)
    mjx_model = mjx.put_model(model)
    mjx_data = mjx.put_data(model, data)
    i_population = jnp.arange(population)
    sin = jnp.asarray(SIN)

    mjx_datas = jax.vmap(lambda _: mjx_data)(i_population)
    step = jax.vmap(jax.jit(mjx.step), in_axes=(None, 0))
    ctrl = jax.vmap(lambda d, i: d.replace(ctrl=sin[i % len(SIN)]))

    with Timer() as t_fst:
        mjx_datas = step(mjx_model, mjx_datas)

    with Timer() as t_rest:
        for i_steps in range(n_steps - 1):
            mjx_datas = step(mjx_model, mjx_datas)
            mjx_datas = ctrl(mjx_datas, i_population)

    return t_fst.elapsed, t_rest.elapsed


def _check_jax_sin(n=1000):
    sin = jnp.sin(jnp.arange(0, 6.3, 0.01))
    irange = jnp.arange(n)
    ctrl = jax.vmap(lambda i: sin[i % len(sin)])
    print(ctrl(irange + 1))


if __name__ == '__main__':
    # _check_jax_sin()
    _gpu_sim_single(100, 100, body_xml=BODIES[0])
    # main_cpu(BODIES[0], 1)


"""
ANT_MP:

        pop=100    pop=200     pop=400     pop=800    pop=1600    pop=3200
100    1.068870   1.982678    3.601267    7.614433   14.172703   27.926051
200    1.996307   3.757835    7.246723   13.981896   25.122914   49.258545
400    3.082119   6.889476   12.237672   23.663918   46.896281   88.551702
800    5.671393  12.298555   21.931903   43.194050   80.859312  158.441200
1600  11.462277  21.579172   43.121553   86.386477  175.034449  364.565426
3200  25.002160  50.309147  101.194015  204.353182  412.987588  830.757445
"""

"""
ANT_NO_MP:

        pop=100    pop=200    pop=400
100    0.393213   0.798133   1.558629
200    0.758073   1.502452   2.960707
400    1.395267   2.818312   5.630566
800    2.824851   5.585441  11.207958
1600   5.917133  11.822327  23.855230
3200  12.820637  25.751082  51.585355
"""