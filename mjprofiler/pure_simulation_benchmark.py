import datetime
import multiprocessing
import time
from concurrent.futures.process import ProcessPoolExecutor as ProcPool

import mujoco
import numpy as np
import pandas as pd

from mjprofiler.bodies import Ant, Humanoid

POPULATION_SIZE = (100, 200, 400, 800, 1600, 3200)
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


def _cpu_sim_single(population: int, n_steps: int, body_xml: str, attempts: int) -> float:
    print(f"{population=} | {n_steps=}")
    model = mujoco.MjModel.from_xml_string(body_xml)

    times = []
    with ProcPool(max_workers=N_PROCS) as pool:
        for _ in range(attempts):
            t = time.perf_counter()
            futures = [pool.submit(_run, model, n_steps, i_population) for i_population in range(population)]
            compute_time = sum([fut.result() for fut in futures])
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
                    attempts
                )
    df = pd.DataFrame({f"{pop=}": results[i] for i, pop in enumerate(POPULATION_SIZE)}, index=list(N_STEPS))
    print(df.to_string())


if __name__ == '__main__':
    main_cpu(BODIES[0], 1)
