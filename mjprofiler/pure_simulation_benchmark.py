import datetime
import multiprocessing
import time
from concurrent.futures.process import ProcessPoolExecutor as ProcPool

import mujoco
import numpy as np
import pandas as pd

from mjprofiler.bodies import Ant, Humanoid

POPULATION_SIZE = (100, 200, 400, 800, 1600, 3200)[:3]
N_STEPS = (100, 200, 400, 800, 1600, 3200)
BODIES = (Ant.make(10, 10, 10), Humanoid.make(10))
SIN = np.sin(np.arange(0, 2 * np.pi, 0.01))
N_PROCS = multiprocessing.cpu_count()


def _run(model, n_steps, i_population):
    _t = time.perf_counter()

    data = mujoco.MjData(model)

    for i_step in range(n_steps):
        mujoco.mj_step(model, data)
        data.ctrl = SIN[(i_population + i_step) % len(SIN)]
    return time.perf_counter() - _t


def _cpu_sim_single(population: int, n_steps: int, body_xml: str, attempts: int, is_mp: bool) -> float:
    print(f"{population=} | {n_steps=}")
    model = mujoco.MjModel.from_xml_string(body_xml)

    times = []
    with ProcPool(max_workers=N_PROCS) as pool:
        for _ in range(attempts):
            t = time.perf_counter()
            if is_mp:
                futures = [pool.submit(_run, model, n_steps, i_population) for i_population in range(population)]
                compute_time = sum([fut.result() for fut in futures])
            else:
                compute_time = sum(_run(model, n_steps, i_population) for i_population in range(population))

            times.append(compute_time)
            total_time = time.perf_counter() - t
            print(f'total: {total_time}, compute: {compute_time}, util: {compute_time / total_time}')

    return sum(times) / len(times)


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


if __name__ == '__main__':
    main_cpu(BODIES[0], 1)

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
