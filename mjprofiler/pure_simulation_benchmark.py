import datetime
import multiprocessing
import time
from concurrent.futures.process import ProcessPoolExecutor as ProcPool
import mujoco
import numpy as np

from mjprofiler.bodies import Ant, Humanoid

POPULATION_SIZE = (100, 200, 400, 800, 1600, 3200)
N_STEPS = (100, 200, 400, 800, 1600, 3200)
BODIES = (Ant.make(10, 10, 10), Humanoid.make(10))
SIN = np.sin(np.arange(0, 2 * np.pi, 0.01))
N_PROCS = multiprocessing.cpu_count()
RUNTIME = [0]


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
            t2 = sum([fut.result() for fut in futures])
            times.append(time.perf_counter() - t)
            print(f'util: {t2 / times[-1]}, n_procs: {N_PROCS} t:{times[-1]}')
            RUNTIME[0] += t2

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

    print(results)
    print(RUNTIME[0])


if __name__ == '__main__':
    main_cpu(BODIES[0], 1)
