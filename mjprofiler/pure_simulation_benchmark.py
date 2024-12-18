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
N_PROCS = multiprocessing.cpu_count()

SIN = np.sin(np.arange(0, 2 * np.pi, 0.01))
SIN_JAX = jnp.asarray(SIN)


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


def main_cpu(body_xml: str):
    print(f"started {datetime.datetime.now()} with {N_PROCS} processes")

    x, y = len(POPULATION_SIZE), len(N_STEPS)
    results = np.zeros((x, y))

    for i_population in range(x):
        for i_steps in range(y):
            results[i_population, i_steps] = _cpu_sim_single(
                    POPULATION_SIZE[i_population],
                    N_STEPS[i_steps],
                    body_xml,
                    is_mp=False
                )
    df = pd.DataFrame({f"{pop=}": results[i] for i, pop in enumerate(POPULATION_SIZE)}, index=list(N_STEPS))
    print(df.to_string())


def _gpu_sim_single(population: int, n_steps: int, body_xml: str, is_ctrl: bool):
    print(f"GPU | {population=} | {n_steps=}")
    model = mujoco.MjModel.from_xml_string(body_xml)
    data = mujoco.MjData(model)
    mjx_model = mjx.put_model(model)
    mjx_data = mjx.put_data(model, data)
    i_population = jnp.arange(population)

    mjx_datas = jax.vmap(lambda _: mjx_data)(i_population)
    step = jax.vmap(jax.jit(mjx.step), in_axes=(None, 0))
    ctrl = jax.vmap(lambda d, i: d.replace(ctrl=SIN_JAX[i % len(SIN)]))

    with Timer() as t_fst:
        mjx_datas = step(mjx_model, mjx_datas)

    with Timer() as t_rest:
        for i_steps in range(n_steps - 1):
            mjx_datas = step(mjx_model, mjx_datas)
            if is_ctrl:
                mjx_datas = ctrl(mjx_datas, i_steps + i_population)

    return t_fst.elapsed, t_rest.elapsed


def main_gpu(body_xml: str):
    print(f"started {datetime.datetime.now()} with {N_PROCS} processes GPU")

    x, y = len(POPULATION_SIZE), len(N_STEPS)
    results = np.zeros((x, y))
    jits = np.zeros((x, y))

    for i_p in range(x):
        for i_s in range(y):
            jits[i_p, i_s], results[i_p, i_s] = _gpu_sim_single(
                    POPULATION_SIZE[i_p],
                    N_STEPS[i_s],
                    body_xml,
                    is_ctrl=False
                )

    for res in jits, results:
        df = pd.DataFrame({f"{pop=}": res[i] for i, pop in enumerate(POPULATION_SIZE)}, index=list(N_STEPS))
        print(df.to_string())


def _check_jit_time(xml: str):
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mjx_model = mjx.put_model(model)
    mjx_data = mjx.put_data(model, data)
    step = jax.jit(mjx.step)

    with Timer() as t:
        step(mjx_model, mjx_data)

    print(f"jit step took {t.elapsed}")
    return t.elapsed


def _check_mp_jit_possible():
    with ProcPool() as pool:
        pool.map(_check_jit_time, [BODIES[0]] * 10)


def check_jit_on_different_size_datas(xml):
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mjx_model = mjx.put_model(model)
    mjx_data = mjx.put_data(model, data)
    step = jax.vmap(jax.jit(mjx.step), in_axes=(None, 0))

    for i in 100, 400, 800:
        with Timer() as t:
            mjx_datas = jax.vmap(lambda _: mjx_data)(jnp.asarray([1] * i))
            step(mjx_model, mjx_datas)
        print(f"fst step with {i} took {t.elapsed}")


if __name__ == '__main__':
    # _check_jax_sin()
    # _gpu_sim_single(100, 100, body_xml=BODIES[0])
    # _check_mp_jit_possible()
    # check_jit_on_different_size_datas(BODIES[0])
    main_gpu(BODIES[0])


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

"""
GPU

First Step
        pop=100    pop=200    pop=400    pop=800   pop=1600   pop=3200
100   26.829207  25.086055  25.827561  26.867411  30.005272  32.846563
200    0.023980   0.025285   0.026983   0.032399   0.052586   0.092934
400    0.023975   0.025294   0.026652   0.032493   0.052658   0.092881
800    0.024145   0.025262   0.027071   0.032330   0.052388   0.094713
1600   0.023801   0.025273   0.026754   0.032108   0.052543   0.092554
3200   0.023968   0.025088   0.027392   0.032736   0.052321   0.092527


1..Nth Step
        pop=100    pop=200    pop=400    pop=800    pop=1600    pop=3200
100    1.989690   2.040878   2.173357   2.640872    4.182794    7.264396
200    3.760479   4.047316   4.253381   5.140228    7.902613   13.633127
400    7.180405   7.733272   8.117275   9.659475   14.492845   24.491088
800   14.546684  15.750672  16.152417  19.434621   29.125457   49.194918
1600  31.879066  33.440370  36.303928  44.913779   73.131966  133.146902
3200  65.761199  70.748011  76.124083  94.294636  160.248001  300.502040


"""