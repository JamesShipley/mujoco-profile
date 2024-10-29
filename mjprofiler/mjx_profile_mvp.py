import time
import multiprocessing
from concurrent.futures.process import ProcessPoolExecutor as ProcPool

import mujoco



_CPU_COUNT = multiprocessing.cpu_count()
_XML = """
    <mujoco model="ant">
      <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
      <option integrator="RK4"/>

      <default>
        <joint armature="1" damping="1" limited="true"/>
        <geom conaffinity="0" condim="3" density="5.0" friction="1 0.5 0.5" margin="0.01" rgba="0.8 0.6 0.4 1"/>
      </default>
      <asset>
        <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0 0 0" type="skybox" width="100"/>
        <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
        <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
        <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
        <material name="geom" texture="texgeom" texuniform="true"/>
      </asset>
      <worldbody>
        <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
        <geom conaffinity="1" condim="3" material="MatPlane" name="floor" pos="0 0 0" rgba="0.8 .9 .8 1" size="40 40 40" type="plane"/>
        <body name="torso" pos="0 0 .75">
          <geom name="torso_geom" pos="0 0 0" size=".25" type="sphere"/>
          <joint armature="0" damping="0" limited="false" margin=".01" name="root" pos="0 0 0" type="free"/>

          <body name="front_left_leg" pos="0 0 0">
            <geom fromto=".0 .0 .0 .2 .2 .0" name="aux_1_geom" size=".08" type="capsule"/>
            <body name="aux_1" pos=".2 .2 0">
              <joint axis="0 0 1" name="hip_1" pos=".0 .0 .0" range="-30 30" type="hinge"/>
              <geom fromto=".0 .0 .0 2.0 2.0 .0" name="left_leg_geom" size=".08" type="capsule"/>
              <body pos="2.0 2.0 0">
                <joint axis="-1 1 0" name="ankle_1" pos=".0 .0 .0" range="30 70" type="hinge"/>
                <geom fromto=".0 .0 .0 2.0 2.0 .0" name="left_ankle_geom" size=".08" type="capsule"/>
              </body>
            </body>
          </body>

          <body name="front_right_leg" pos="0 0 0">
            <geom fromto=".0 .0 .0 -.2 .2 .0" name="aux_2_geom" size=".08" type="capsule"/>
            <body name="aux_2" pos="-.2 .2 0">
              <joint axis="0 0 1" name="hip_2" pos=".0 .0 .0" range="-30 30" type="hinge"/>
              <geom fromto=".0 .0 .0 -2.0 2.0 .0" name="right_leg_geom" size=".08" type="capsule"/>
              <body pos="-2.0 2.0 0">
                <joint axis="1 1 0" name="ankle_2" pos=".0 .0 .0" range="-70 -30" type="hinge"/>
                <geom fromto=".0 .0 .0 -2.0 2.0 .0" name="right_ankle_geom" size=".08" type="capsule"/>
              </body>
            </body>
          </body>

          <body name="back_leg" pos="0 0 0">
            <geom fromto=".0 .0 .0 -.2 -.2 .0" name="aux_3_geom" size=".08" type="capsule"/>
            <body name="aux_3" pos="-.2 -.2 0">
              <joint axis="0 0 1" name="hip_3" pos=".0 .0 .0" range="-30 30" type="hinge"/>
              <geom fromto=".0 .0 .0 -2.0 -2.0 .0" name="back_leg_geom" size=".08" type="capsule"/>
              <body pos="-2.0 -2.0 0">
                <joint axis="-1 1 0" name="ankle_3" pos=".0 .0 .0" range="-70 -30" type="hinge"/>
                <geom fromto=".0 .0 .0 -2.0 -2.0 .0" name="third_ankle_geom" size=".08" type="capsule"/>
              </body>
            </body>
          </body>

          <body name="right_back_leg" pos="0 0 0">
            <geom fromto=".0 .0 .0 .2 -.2 .0" name="aux_4_geom" size=".08" type="capsule"/>
            <body name="aux_4" pos=".2 -.2 0">
            <joint axis="0 0 1" name="hip_4" pos=".0 .0 .0" range="-30 30" type="hinge"/>
              <geom fromto=".0 .0 .0 2.0 -2.0 .0" name="rightback_leg_geom" size=".08" type="capsule"/>
              <body pos="2.0 -2.0 0">
                <joint axis="1 1 0" name="ankle_4" pos=".0 .0 .0" range="30 70" type="hinge"/>
                <geom fromto=".0 .0 .0 2.0 -2.0 .0" name="fourth_ankle_geom" size=".08" type="capsule"/>
              </body>
            </body>
          </body>

        </body>
      </worldbody>

      <actuator>
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_1" gear="150"/>
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_1" gear="150"/>
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_2" gear="150"/>
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_2" gear="150"/>
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_3" gear="150"/>
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_3" gear="150"/>
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_4" gear="150"/>
        <motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_4" gear="150"/>
      </actuator>
    </mujoco>
    """


def _cpu_profile_inner(model_xml: str, n_steps: int):
    model = mujoco.MjModel.from_xml_string(model_xml)
    data = mujoco.MjData(model)
    data.ctrl = 1

    for i_step in range(n_steps):
        mujoco.mj_step(model, data)


def cpu_profile(model_xml: str, n_variants: int, n_steps: int, max_processes: int):
    print(f"running CPU profile with {n_variants=}, {n_steps=}, {max_processes=} ...")
    assert 0 < max_processes <= _CPU_COUNT

    t = time.perf_counter()

    with ProcPool(max_workers=max_processes) as pool:
        futures = [pool.submit(_cpu_profile_inner, model_xml, n_steps) for _ in range(n_variants)]
        _ = [fut.result() for fut in futures]

    return time.perf_counter() - t


def gpu_profile(model_xml: str, n_variants: int, n_steps: int):
    print(f"running GPU profile with {n_variants=}, {n_steps=} ...")
    from mujoco import mjx
    import jax
    import jax.numpy as jnp
    model = mujoco.MjModel.from_xml_string(model_xml)
    data = mujoco.MjData(model)
    mjx_model = mjx.put_model(model)
    mjx_data = mjx.put_data(model, data)
    mjx_datas = jax.vmap(lambda _: mjx_data.replace(ctrl=1))(jnp.arange(n_variants))
    step = jax.vmap(jax.jit(mjx.step), in_axes=(None, 0))

    # do not even time first step, takes too long, as long as step 2...N is faster we have viable option for GPU
    # because first step is fixed cost
    mjx_datas = step(mjx_model, mjx_datas)
    t = time.perf_counter()

    for i_steps in range(n_steps - 1):
        mjx_datas = step(mjx_model, mjx_datas)

    return time.perf_counter() - t


def compare(model_xml: str, n_variants: int, n_steps: int, max_processes: int):
    time_cpu = cpu_profile(model_xml, n_variants, n_steps, max_processes)
    time_gpu = gpu_profile(model_xml, n_variants, n_steps)
    return time_cpu > time_gpu


def main(max_processes: int | None = None):
    if max_processes is None:
        max_processes = _CPU_COUNT

    variants = [400, 800, 1600]
    steps = [1000, 2000, 4000, 8000]

    cpus = {
        (n_variants, n_steps): cpu_profile(_XML, n_variants, n_steps, max_processes)
        for n_variants in variants
        for n_steps in steps
    }

    gpus = {
        (n_variants, n_steps): gpu_profile(_XML, n_variants, n_steps)
        for n_variants in variants
        for n_steps in steps
    }

    print("=" * 80)
    for (n_variants, n_steps) in cpus:
        cpu = cpus[n_variants, n_steps]
        gpu = gpus[n_variants, n_steps]
        gpu_better = ("worse", "better")[gpu < cpu]

        faster, slower = (cpu, gpu)[::2 * (gpu > cpu) - 1]
        percentage = int(100 * (slower / faster - 1))
        print(f"({n_variants}, {n_steps}): GPU {gpu_better} | {cpu=:.3f} {gpu=:.3f} | {percentage}%")

    print("=" * 80)
    if not any(cpus[k] > gpus[k] for k in cpus):
        print("GPU is literally never better")


if __name__ == '__main__':
    main()