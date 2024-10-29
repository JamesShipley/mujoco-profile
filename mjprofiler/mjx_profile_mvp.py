import time
import multiprocessing
from concurrent.futures.process import ProcessPoolExecutor as ProcPool

import mujoco
from mujoco import mjx
import jax
import jax.numpy as jnp


_CPU_COUNT = multiprocessing.cpu_count()


def _cpu_profile_inner(model_xml: str, n_steps: int):
    model = mujoco.MjModel.from_xml_string(model_xml)
    data = mujoco.MjData(model)
    data.ctrl = 1

    for i_step in range(n_steps):
        mujoco.mj_step(model, data)


def cpu_profile(model_xml: str, n_variants: int, n_steps: int, max_processes: int | None = None):
    if max_processes is None:
        max_processes = _CPU_COUNT

    assert 0 < max_processes <= _CPU_COUNT

    t = time.perf_counter()

    with ProcPool(max_workers=max_processes) as pool:
        futures = [pool.submit(_cpu_profile_inner, model_xml, n_steps) for _ in range(n_variants)]
        _ = [fut.result() for fut in futures]

    return time.perf_counter() - t


def gpu_profile(model_xml: str, n_variants: int, n_steps: int):
    model = mujoco.MjModel.from_xml_string(model_xml)
    data = mujoco.MjData(model)
    mjx_model = mjx.put_model(model)
    mjx_data = mjx.put_data(model, data)
    mjx_datas = jax.vmap(lambda _: mjx_data)(jnp.arange(n_variants))
    step = jax.vmap(jax.jit(mjx.step), in_axes=(None, 0))

    # do not even time first step, takes too long, as long as step 2...N is faster we have
    mjx_datas = step(mjx_model, mjx_datas)
    t = time.perf_counter()

    for i_steps in range(n_steps - 1):
        mjx_datas = step(mjx_model, mjx_datas)

    return time.perf_counter() - t


def compare(model_xml: str, n_variants: int, n_steps: int, max_processes: int | None = None):

    print(f"running CPU profile with {n_variants=}, {n_steps=}, {max_processes=} ...")
    time_cpu = cpu_profile(model_xml, n_variants, n_steps, max_processes)
    print(f"finished CPU profile (took {time_cpu:.3f} seconds), running GPU profile ...")
    time_gpu = gpu_profile(model_xml, n_variants, n_steps)
    print(f"finished both, {time_gpu=}, {time_cpu=}, cpu {['slower','faster'][time_cpu < time_gpu]}")
