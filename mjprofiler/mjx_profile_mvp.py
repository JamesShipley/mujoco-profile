import time
import multiprocessing
from concurrent.futures.process import ProcessPoolExecutor as ProcPool

import mujoco



_CPU_COUNT = multiprocessing.cpu_count()
_XML_ANT = """
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

_XML_BALL = """
<mujoco>
  <worldbody>
    <light name="top" pos="0 0 1"/>
    <body name="box_and_sphere" euler="0 0 -30">
      <joint name="swing" type="hinge" axis="1 -1 0" pos="-.2 -.2 -.2"/>
      <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
      <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""

_XML_HUMANOID = """
<mujoco model="Humanoid">
  <option timestep="0.005" iterations="1" ls_iterations="4">
    <flag eulerdamp="disable"/>
  </option>

  <visual>
    <map force="0.1" zfar="30"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global offwidth="2560" offheight="1440" elevation="-20" azimuth="120"/>
  </visual>

  <statistic center="0 0 0.7"/>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".3 .5 .7" rgb2="0 0 0" width="32" height="512"/>
    <texture name="body" type="cube" builtin="flat" mark="cross" width="128" height="128" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" markrgb="1 1 1" random="0.01"/>
    <material name="body" texture="body" texuniform="true" rgba="0.8 0.6 .4 1"/>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
  </asset>

  <default>
    <motor ctrlrange="-1 1" ctrllimited="true"/>
    <default class="body">

      <!-- geoms -->
      <!-- TODO(robotics-simulation): support condim=1 for humanoid capsules. -->
      <geom type="capsule" condim="3" friction=".7" solimp=".9 .99 .003" solref=".015 1" material="body" contype="0" conaffinity="0"/>
      <default class="thigh">
        <geom size=".06"/>
      </default>
      <default class="shin">
        <geom fromto="0 0 0 0 0 -.3"  size=".049"/>
      </default>
      <default class="foot">
        <geom size=".027"/>
        <default class="foot1">
          <geom fromto="-.07 -.01 0 .14 -.03 0"/>
        </default>
        <default class="foot2">
          <geom fromto="-.07 .01 0 .14  .03 0"/>
        </default>
      </default>
      <default class="arm_upper">
        <geom size=".04"/>
      </default>
      <default class="arm_lower">
        <geom size=".031"/>
      </default>
      <default class="hand">
        <geom type="sphere" size=".04"/>
      </default>

      <!-- joints -->
      <joint type="hinge" damping=".2" stiffness="1" armature=".01" limited="true" solimplimit="0 .99 .01"/>
      <default class="joint_big">
        <joint damping="5" stiffness="10"/>
        <default class="hip_x">
          <joint range="-30 10"/>
        </default>
        <default class="hip_z">
          <joint range="-60 35"/>
        </default>
        <default class="hip_y">
          <joint axis="0 1 0" range="-150 20"/>
        </default>
        <default class="joint_big_stiff">
          <joint stiffness="20"/>
        </default>
      </default>
      <default class="knee">
        <joint pos="0 0 .02" axis="0 -1 0" range="-160 2"/>
      </default>
      <default class="ankle">
        <joint range="-50 50"/>
        <default class="ankle_y">
          <joint pos="0 0 .08" axis="0 1 0" stiffness="6"/>
        </default>
        <default class="ankle_x">
          <joint pos="0 0 .04" stiffness="3"/>
        </default>
      </default>
      <default class="shoulder">
        <joint range="-85 60"/>
      </default>
      <default class="elbow">
        <joint range="-100 50" stiffness="0"/>
      </default>
    </default>
  </default>

  <worldbody>
    <geom name="floor" size="0 0 .05" type="plane" material="grid" condim="3"/>
    <light name="spotlight" mode="targetbodycom" target="torso" diffuse=".8 .8 .8" specular="0.3 0.3 0.3" pos="0 -6 4" cutoff="30"/>
    <body name="torso" pos="0 0 1.282" childclass="body">
      <light name="top" pos="0 0 2" mode="trackcom"/>
      <freejoint name="root"/>
      <geom name="torso" fromto="0 -.07 0 0 .07 0" size=".07"/>
      <geom name="waist_upper" fromto="-.01 -.06 -.12 -.01 .06 -.12" size=".06"/>
      <body name="head" pos="0 0 .19">
        <geom name="head" type="sphere" size=".09"/>
      </body>
      <body name="waist_lower" pos="-.01 0 -.26">
        <geom name="waist_lower" fromto="0 -.06 0 0 .06 0" size=".06"/>
        <joint name="abdomen_z" pos="0 0 .065" axis="0 0 1" range="-45 45" class="joint_big_stiff"/>
        <joint name="abdomen_y" pos="0 0 .065" axis="0 1 0" range="-75 30" class="joint_big"/>
        <body name="pelvis" pos="0 0 -.165">
          <joint name="abdomen_x" pos="0 0 .1" axis="1 0 0" range="-35 35" class="joint_big"/>
          <geom name="butt" fromto="-.02 -.07 0 -.02 .07 0" size=".09"/>
          <body name="thigh_right" pos="0 -.1 -.04">
            <joint name="hip_x_right" axis="1 0 0" class="hip_x"/>
            <joint name="hip_z_right" axis="0 0 1" class="hip_z"/>
            <joint name="hip_y_right" class="hip_y"/>
            <geom name="thigh_right" fromto="0 0 0 0 .01 -.34" class="thigh"/>
            <body name="shin_right" pos="0 .01 -.4">
              <joint name="knee_right" class="knee"/>
              <geom name="shin_right" class="shin"/>
              <body name="foot_right" pos="0 0 -.39">
                <joint name="ankle_y_right" class="ankle_y"/>
                <joint name="ankle_x_right" class="ankle_x" axis="1 0 .5"/>
                <geom name="foot1_right" class="foot1"/>
                <geom name="foot2_right" class="foot2"/>
              </body>
            </body>
          </body>
          <body name="thigh_left" pos="0 .1 -.04">
            <joint name="hip_x_left" axis="-1 0 0" class="hip_x"/>
            <joint name="hip_z_left" axis="0 0 -1" class="hip_z"/>
            <joint name="hip_y_left" class="hip_y"/>
            <geom name="thigh_left" fromto="0 0 0 0 -.01 -.34" class="thigh"/>
            <body name="shin_left" pos="0 -.01 -.4">
              <joint name="knee_left" class="knee"/>
              <geom name="shin_left" fromto="0 0 0 0 0 -.3" class="shin"/>
              <body name="foot_left" pos="0 0 -.39">
                <joint name="ankle_y_left" class="ankle_y"/>
                <joint name="ankle_x_left" class="ankle_x" axis="-1 0 -.5"/>
                <geom name="foot1_left" class="foot1"/>
                <geom name="foot2_left" class="foot2"/>
              </body>
            </body>
          </body>
        </body>
      </body>
      <body name="upper_arm_right" pos="0 -.17 .06">
        <joint name="shoulder1_right" axis="2 1 1"  class="shoulder"/>
        <joint name="shoulder2_right" axis="0 -1 1" class="shoulder"/>
        <geom name="upper_arm_right" fromto="0 0 0 .16 -.16 -.16" class="arm_upper"/>
        <body name="lower_arm_right" pos=".18 -.18 -.18">
          <joint name="elbow_right" axis="0 -1 1" class="elbow"/>
          <geom name="lower_arm_right" fromto=".01 .01 .01 .17 .17 .17" class="arm_lower"/>
          <body name="hand_right" pos=".18 .18 .18">
            <geom name="hand_right" zaxis="1 1 1" class="hand"/>
          </body>
        </body>
      </body>
      <body name="upper_arm_left" pos="0 .17 .06">
        <joint name="shoulder1_left" axis="-2 1 -1" class="shoulder"/>
        <joint name="shoulder2_left" axis="0 -1 -1"  class="shoulder"/>
        <geom name="upper_arm_left" fromto="0 0 0 .16 .16 -.16" class="arm_upper"/>
        <body name="lower_arm_left" pos=".18 .18 -.18">
          <joint name="elbow_left" axis="0 -1 -1" class="elbow"/>
          <geom name="lower_arm_left" fromto=".01 -.01 .01 .17 -.17 .17" class="arm_lower"/>
          <body name="hand_left" pos=".18 -.18 .18">
            <geom name="hand_left" zaxis="1 -1 1" class="hand"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>

  <contact>
    <exclude body1="waist_lower" body2="thigh_right"/>
    <exclude body1="waist_lower" body2="thigh_left"/>
    <pair geom1="foot1_left" geom2="floor"/>
    <pair geom1="foot1_right" geom2="floor"/>
    <pair geom1="foot2_left" geom2="floor"/>
    <pair geom1="foot2_right" geom2="floor"/>
  </contact>

  <!-- <tendon>
    <fixed name="hamstring_right" limited="true" range="-0.3 2">
      <joint joint="hip_y_right" coef=".5"/>
      <joint joint="knee_right" coef="-.5"/>
    </fixed>
    <fixed name="hamstring_left" limited="true" range="-0.3 2">
      <joint joint="hip_y_left" coef=".5"/>
      <joint joint="knee_left" coef="-.5"/>
    </fixed>
  </tendon> -->

  <actuator>
    <motor name="abdomen_y"       gear="40"  joint="abdomen_y"/>
    <motor name="abdomen_z"       gear="40"  joint="abdomen_z"/>
    <motor name="abdomen_x"       gear="40"  joint="abdomen_x"/>
    <motor name="hip_x_right"     gear="40"  joint="hip_x_right"/>
    <motor name="hip_z_right"     gear="40"  joint="hip_z_right"/>
    <motor name="hip_y_right"     gear="120" joint="hip_y_right"/>
    <motor name="knee_right"      gear="80"  joint="knee_right"/>
    <motor name="ankle_x_right"   gear="20"  joint="ankle_x_right"/>
    <motor name="ankle_y_right"   gear="20"  joint="ankle_y_right"/>
    <motor name="hip_x_left"      gear="40"  joint="hip_x_left"/>
    <motor name="hip_z_left"      gear="40"  joint="hip_z_left"/>
    <motor name="hip_y_left"      gear="120" joint="hip_y_left"/>
    <motor name="knee_left"       gear="80"  joint="knee_left"/>
    <motor name="ankle_x_left"    gear="20"  joint="ankle_x_left"/>
    <motor name="ankle_y_left"    gear="20"  joint="ankle_y_left"/>
    <motor name="shoulder1_right" gear="20"  joint="shoulder1_right"/>
    <motor name="shoulder2_right" gear="20"  joint="shoulder2_right"/>
    <motor name="elbow_right"     gear="40"  joint="elbow_right"/>
    <motor name="shoulder1_left"  gear="20"  joint="shoulder1_left"/>
    <motor name="shoulder2_left"  gear="20"  joint="shoulder2_left"/>
    <motor name="elbow_left"      gear="40"  joint="elbow_left"/>
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
    cpu = cpu_profile(model_xml, n_variants, n_steps, max_processes)
    gpu = gpu_profile(model_xml, n_variants, n_steps)
    gpu_win = ("worse", "better")[gpu < cpu]
    faster, slower = (cpu, gpu)[::2 * (gpu > cpu) - 1]
    percentage = int(100 * (slower / faster - 1))
    print(f"({n_variants}, {n_steps}): GPU {gpu_win} | {cpu=:.3f} {gpu=:.3f} | {percentage}%")


def main(xml=_XML_HUMANOID, max_processes: int | None = None):
    if max_processes is None:
        max_processes = _CPU_COUNT

    variants = [100, 1000, 4000]
    steps = [100, 500, 1000]

    cpus = {
        (n_variants, n_steps): cpu_profile(xml, n_variants, n_steps, max_processes)
        for n_variants in variants
        for n_steps in steps
    }

    gpus = {
        (n_variants, n_steps): gpu_profile(xml, n_variants, n_steps)
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
        print(f"{n_variants=} {n_steps=} | GPU {gpu_better} | {cpu=:.3f} {gpu=:.3f} | {percentage}%")

    print("=" * 80)
    if not any(cpus[k] > gpus[k] for k in cpus):
        print("GPU is literally never better")


if __name__ == '__main__':
    main(_XML_HUMANOID)