ROBOT_XML = """
<mujoco model="scene">
  <compiler autolimits="true" angle="radian"/>
  <option timestep="0.001" gravity="0 0 -9.8100000000000005" integrator="RK4"/>
  <visual>
    <headlight active="0"/>
  </visual>
  <default>
    <default class="/"/>
    <default class="mbs0/"/>
    <default class="mbs1/"/>
  </default>
  <asset>
    <material name="heightmap_0_material" class="/" texrepeat="1 1" emission="0" specular="0.5" shininess="0.5" reflectance="0" rgba="0.39215686274509803 0.39215686274509803 0.39215686274509803 1"/>
    <material name="mbs1/geom_mbs1_geom0_material" class="mbs1/" texrepeat="1 1" emission="0" specular="0.5" shininess="0.5" reflectance="0" rgba="1 0.19607843137254902 0.19607843137254902 1"/>
    <material name="mbs1/geom_mbs1_geom1_material" class="mbs1/" texrepeat="1 1" emission="0" specular="0.5" shininess="0.5" reflectance="0" rgba="1 1 1 1"/>
    <material name="mbs1/geom_mbs1_geom2_material" class="mbs1/" texrepeat="1 1" emission="0" specular="0.5" shininess="0.5" reflectance="0" rgba="1 1 1 1"/>
    <material name="mbs1/geom_mbs1_geom3_material" class="mbs1/" texrepeat="1 1" emission="0" specular="0.5" shininess="0.5" reflectance="0" rgba="1 1 1 1"/>
    <material name="mbs1/geom_mbs1_link0_geom0_material" class="mbs1/" texrepeat="1 1" emission="0" specular="0.5" shininess="0.5" reflectance="0" rgba="1 1 1 1"/>
    <material name="mbs1/geom_mbs1_link0_geom1_material" class="mbs1/" texrepeat="1 1" emission="0" specular="0.5" shininess="0.5" reflectance="0" rgba="1 1 1 1"/>
    <material name="mbs1/geom_mbs1_link0_geom2_material" class="mbs1/" texrepeat="1 1" emission="0" specular="0.5" shininess="0.5" reflectance="0" rgba="0.19607843137254902 0.19607843137254902 1 1"/>
    <material name="mbs1/geom_mbs1_link0_geom3_material" class="mbs1/" texrepeat="1 1" emission="0" specular="0.5" shininess="0.5" reflectance="0" rgba="1 1 1 1"/>
    <material name="mbs1/geom_mbs1_link0_link1_geom0_material" class="mbs1/" texrepeat="1 1" emission="0" specular="0.5" shininess="0.5" reflectance="0" rgba="1 1 1 1"/>
    <material name="mbs1/geom_mbs1_link0_link1_geom1_material" class="mbs1/" texrepeat="1 1" emission="0" specular="0.5" shininess="0.5" reflectance="0" rgba="1 1 1 1"/>
    <material name="mbs1/geom_mbs1_link0_link1_geom2_material" class="mbs1/" texrepeat="1 1" emission="0" specular="0.5" shininess="0.5" reflectance="0" rgba="0.19607843137254902 0.19607843137254902 1 1"/>
    <material name="mbs1/geom_mbs1_link0_link1_geom3_material" class="mbs1/" texrepeat="1 1" emission="0" specular="0.5" shininess="0.5" reflectance="0" rgba="1 1 1 1"/>
    <material name="mbs1/geom_mbs1_link0_link1_geom4_material" class="mbs1/" texrepeat="1 1" emission="0" specular="0.5" shininess="0.5" reflectance="0" rgba="1 1 1 1"/>
    <material name="mbs1/geom_mbs1_link0_link1_link1_geom0_material" class="mbs1/" texrepeat="1 1" emission="0" specular="0.5" shininess="0.5" reflectance="0" rgba="1 1 1 1"/>
    <material name="mbs1/geom_mbs1_link0_link1_link1_geom1_material" class="mbs1/" texrepeat="1 1" emission="0" specular="0.5" shininess="0.5" reflectance="0" rgba="1 1 1 1"/>
    <material name="mbs1/geom_mbs1_link0_link1_link1_geom2_material" class="mbs1/" texrepeat="1 1" emission="0" specular="0.5" shininess="0.5" reflectance="0" rgba="0.19607843137254902 0.19607843137254902 1 1"/>
    <material name="mbs1/geom_mbs1_link0_link1_link2_geom0_material" class="mbs1/" texrepeat="1 1" emission="0" specular="0.5" shininess="0.5" reflectance="0" rgba="1 1 1 1"/>
    <material name="mbs1/geom_mbs1_link0_link1_link2_geom1_material" class="mbs1/" texrepeat="1 1" emission="0" specular="0.5" shininess="0.5" reflectance="0" rgba="1 1 1 1"/>
    <material name="mbs1/geom_mbs1_link0_link1_link2_geom2_material" class="mbs1/" texrepeat="1 1" emission="0" specular="0.5" shininess="0.5" reflectance="0" rgba="0.19607843137254902 0.19607843137254902 1 1"/>
    <material name="mbs1/geom_mbs1_link1_geom0_material" class="mbs1/" texrepeat="1 1" emission="0" specular="0.5" shininess="0.5" reflectance="0" rgba="1 1 1 1"/>
    <material name="mbs1/geom_mbs1_link1_geom1_material" class="mbs1/" texrepeat="1 1" emission="0" specular="0.5" shininess="0.5" reflectance="0" rgba="1 1 1 1"/>
    <material name="mbs1/geom_mbs1_link1_geom2_material" class="mbs1/" texrepeat="1 1" emission="0" specular="0.5" shininess="0.5" reflectance="0" rgba="0.19607843137254902 0.19607843137254902 1 1"/>
    <material name="mbs1/geom_mbs1_link2_geom0_material" class="mbs1/" texrepeat="1 1" emission="0" specular="0.5" shininess="0.5" reflectance="0" rgba="1 1 1 1"/>
    <material name="mbs1/geom_mbs1_link2_geom1_material" class="mbs1/" texrepeat="1 1" emission="0" specular="0.5" shininess="0.5" reflectance="0" rgba="1 1 1 1"/>
    <material name="mbs1/geom_mbs1_link2_geom2_material" class="mbs1/" texrepeat="1 1" emission="0" specular="0.5" shininess="0.5" reflectance="0" rgba="0.19607843137254902 0.19607843137254902 1 1"/>
  </asset>
  <worldbody>
    <light name="//unnamed_light_0" class="/" directional="true" castshadow="false" pos="0 0 100" ambient="0.5 0.5 0.5"/>
    <body pos="0 0 0" quat="0 0 0 1" name="mbs0/"/>
    <geom name="//unnamed_geom_0" class="/" type="plane" size="10 10 1" material="heightmap_0_material" pos="0 0 0" quat="0 0 0 1"/>
    <body pos="0 0 0.030150000000000027" quat="0 0 0 1" name="mbs1/">
      <freejoint name="mbs1/"/>
      <geom name="mbs1/mbs1_geom0" class="mbs1/" type="box" size="0.044499999999999998 0.044499999999999998 0.03015" material="mbs1/geom_mbs1_geom0_material"/>
      <geom name="mbs1/mbs1_geom1" class="mbs1/" type="box" size="0.0089999999999999993 0.026499999999999999 0.0082945499999999995" material="mbs1/geom_mbs1_geom1_material" pos="-0.053499999999999999 0 0" quat="0 0 0.70710700000000004 0.70710700000000004"/>
      <geom name="mbs1/mbs1_geom2" class="mbs1/" type="box" size="0.0089999999999999993 0.026499999999999999 0.0082945499999999995" material="mbs1/geom_mbs1_geom2_material" pos="0 0.053499999999999999 0" quat="0.70710700000000004 0 0 0.70710700000000004"/>
      <geom name="mbs1/mbs1_geom3" class="mbs1/" type="box" size="0.0089999999999999993 0.026499999999999999 0.0082945499999999995" material="mbs1/geom_mbs1_geom3_material" pos="0 -0.053499999999999999 0" quat="0.70710700000000004 0 0 -0.70710700000000004"/>
      <body name="mbs1/mbs1_link0" pos="-0.079024999999999998 0 0" quat="0 0 0.70710700000000004 0.70710700000000004">
        <inertial pos="0.048371400000000002 0 0" quat="0 0.70710700000000004 0 0.70710700000000004" mass="0.11899999999999999" diaginertia="0.00019916199999999999 0.00018530599999999999 4.57711e-05"/>
        <joint name="mbs1/mbs1_joint0" class="mbs1/" pos="0 0 0" axis="0 1 0" range="-1.0471999999999999 1.0471999999999999" armature="0.002"/>
        <geom name="mbs1/mbs1_link0_geom0" class="mbs1/" type="box" size="0.029149999999999999 0.025600000000000001 0.01" material="mbs1/geom_mbs1_link0_geom0_material" pos="0.017999999999999999 0 0"/>
        <geom name="mbs1/mbs1_link0_geom1" class="mbs1/" type="box" size="0.001 0.026499999999999999 0.026499999999999999" material="mbs1/geom_mbs1_link0_geom1_material" pos="0.048149999999999998 0 0"/>
        <geom name="mbs1/mbs1_link0_geom2" class="mbs1/" type="box" size="0.031443100000000002 0.031443100000000002 0.03015" material="mbs1/geom_mbs1_link0_geom2_material" pos="0.080593100000000001 0 0" quat="0.70710700000000004 -0.70710700000000004 0 0"/>
        <geom name="mbs1/mbs1_link0_geom3" class="mbs1/" type="box" size="0.0089999999999999993 0.026499999999999999 0.0082945499999999995" material="mbs1/geom_mbs1_link0_geom3_material" pos="0.121036 0 0"/>
        <body name="mbs1/mbs1_link0_link1" pos="0.146561 0 0">
          <inertial pos="0.047675700000000001 0 0" quat="0.5 0.5 0.5 0.5" mass="0.13" diaginertia="0.00019109199999999999 0.00016208499999999999 8.4671899999999998e-05"/>
          <joint name="mbs1/mbs1_link0_joint1" class="mbs1/" pos="0 0 0" axis="0 1 0" range="-1.0471999999999999 1.0471999999999999" armature="0.002"/>
          <geom name="mbs1/mbs1_link0_link1_geom0" class="mbs1/" type="box" size="0.029149999999999999 0.025600000000000001 0.01" material="mbs1/geom_mbs1_link0_link1_geom0_material" pos="0.017999999999999999 0 0"/>
          <geom name="mbs1/mbs1_link0_link1_geom1" class="mbs1/" type="box" size="0.001 0.026499999999999999 0.026499999999999999" material="mbs1/geom_mbs1_link0_link1_geom1_material" pos="0.048149999999999998 0 0"/>
          <geom name="mbs1/mbs1_link0_link1_geom2" class="mbs1/" type="box" size="0.031443100000000002 0.031443100000000002 0.03015" material="mbs1/geom_mbs1_link0_link1_geom2_material" pos="0.080593100000000001 0 0" quat="0.70710700000000004 -0.70710700000000004 0 0"/>
          <geom name="mbs1/mbs1_link0_link1_geom3" class="mbs1/" type="box" size="0.0089999999999999993 0.026499999999999999 0.0082945499999999995" material="mbs1/geom_mbs1_link0_link1_geom3_material" pos="0.080593100000000001 0 -0.040443100000000003" quat="0.5 -0.5 0.5 0.5"/>
          <geom name="mbs1/mbs1_link0_link1_geom4" class="mbs1/" type="box" size="0.0089999999999999993 0.026499999999999999 0.0082945499999999995" material="mbs1/geom_mbs1_link0_link1_geom4_material" pos="0.080593100000000001 0 0.040443100000000003" quat="0.5 -0.5 -0.5 -0.5"/>
          <body name="mbs1/mbs1_link0_link1_link1" pos="0.080593100000000001 0 -0.065968100000000002" quat="0.5 -0.5 0.5 0.5">
            <inertial pos="0.040970300000000001 0 0" quat="0 0.70710700000000004 0 0.70710700000000004" mass="0.108" diaginertia="0.00013229299999999999 0.000120759 4.2943900000000001e-05"/>
            <joint name="mbs1/mbs1_link0_link1_joint1" class="mbs1/" pos="0 0 0" axis="0 1 0" range="-1.0471999999999999 1.0471999999999999" armature="0.002"/>
            <geom name="mbs1/mbs1_link0_link1_link1_geom0" class="mbs1/" type="box" size="0.029149999999999999 0.025600000000000001 0.01" material="mbs1/geom_mbs1_link0_link1_link1_geom0_material" pos="0.017999999999999999 0 0"/>
            <geom name="mbs1/mbs1_link0_link1_link1_geom1" class="mbs1/" type="box" size="0.001 0.026499999999999999 0.026499999999999999" material="mbs1/geom_mbs1_link0_link1_link1_geom1_material" pos="0.048149999999999998 0 0"/>
            <geom name="mbs1/mbs1_link0_link1_link1_geom2" class="mbs1/" type="box" size="0.031443100000000002 0.031443100000000002 0.03015" material="mbs1/geom_mbs1_link0_link1_link1_geom2_material" pos="0.080593100000000001 0 0"/>
          </body>
          <body name="mbs1/mbs1_link0_link1_link2" pos="0.080593100000000001 0 0.065968100000000002" quat="0.5 -0.5 -0.5 -0.5">
            <inertial pos="0.040970300000000001 0 0" quat="0 0.70710700000000004 0 0.70710700000000004" mass="0.108" diaginertia="0.00013229299999999999 0.000120759 4.2943900000000001e-05"/>
            <joint name="mbs1/mbs1_link0_link1_joint2" class="mbs1/" pos="0 0 0" axis="0 1 0" range="-1.0471999999999999 1.0471999999999999" armature="0.002"/>
            <geom name="mbs1/mbs1_link0_link1_link2_geom0" class="mbs1/" type="box" size="0.029149999999999999 0.025600000000000001 0.01" material="mbs1/geom_mbs1_link0_link1_link2_geom0_material" pos="0.017999999999999999 0 0"/>
            <geom name="mbs1/mbs1_link0_link1_link2_geom1" class="mbs1/" type="box" size="0.001 0.026499999999999999 0.026499999999999999" material="mbs1/geom_mbs1_link0_link1_link2_geom1_material" pos="0.048149999999999998 0 0"/>
            <geom name="mbs1/mbs1_link0_link1_link2_geom2" class="mbs1/" type="box" size="0.031443100000000002 0.031443100000000002 0.03015" material="mbs1/geom_mbs1_link0_link1_link2_geom2_material" pos="0.080593100000000001 0 0"/>
          </body>
        </body>
      </body>
      <body name="mbs1/mbs1_link1" pos="0 0.079024999999999998 0" quat="0.70710700000000004 0 0 0.70710700000000004">
        <inertial pos="0.040970300000000001 0 0" quat="0 0.70710700000000004 0 0.70710700000000004" mass="0.108" diaginertia="0.00013229299999999999 0.000120759 4.2943900000000001e-05"/>
        <joint name="mbs1/mbs1_joint1" class="mbs1/" pos="0 0 0" axis="0 1 0" range="-1.0471999999999999 1.0471999999999999" armature="0.002"/>
        <geom name="mbs1/mbs1_link1_geom0" class="mbs1/" type="box" size="0.029149999999999999 0.025600000000000001 0.01" material="mbs1/geom_mbs1_link1_geom0_material" pos="0.017999999999999999 0 0"/>
        <geom name="mbs1/mbs1_link1_geom1" class="mbs1/" type="box" size="0.001 0.026499999999999999 0.026499999999999999" material="mbs1/geom_mbs1_link1_geom1_material" pos="0.048149999999999998 0 0"/>
        <geom name="mbs1/mbs1_link1_geom2" class="mbs1/" type="box" size="0.031443100000000002 0.031443100000000002 0.03015" material="mbs1/geom_mbs1_link1_geom2_material" pos="0.080593100000000001 0 0"/>
      </body>
      <body name="mbs1/mbs1_link2" pos="0 -0.079024999999999998 0" quat="0.70710700000000004 0 0 -0.70710700000000004">
        <inertial pos="0.040970300000000001 0 0" quat="0 0.70710700000000004 0 0.70710700000000004" mass="0.108" diaginertia="0.00013229299999999999 0.000120759 4.2943900000000001e-05"/>
        <joint name="mbs1/mbs1_joint2" class="mbs1/" pos="0 0 0" axis="0 1 0" range="-1.0471999999999999 1.0471999999999999" armature="0.002"/>
        <geom name="mbs1/mbs1_link2_geom0" class="mbs1/" type="box" size="0.029149999999999999 0.025600000000000001 0.01" material="mbs1/geom_mbs1_link2_geom0_material" pos="0.017999999999999999 0 0"/>
        <geom name="mbs1/mbs1_link2_geom1" class="mbs1/" type="box" size="0.001 0.026499999999999999 0.026499999999999999" material="mbs1/geom_mbs1_link2_geom1_material" pos="0.048149999999999998 0 0"/>
        <geom name="mbs1/mbs1_link2_geom2" class="mbs1/" type="box" size="0.031443100000000002 0.031443100000000002 0.03015" material="mbs1/geom_mbs1_link2_geom2_material" pos="0.080593100000000001 0 0"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="mbs1/actuator_position_mbs1_joint0" class="mbs1/" joint="mbs1/mbs1_joint0" kp="5"/>
    <velocity name="mbs1/actuator_velocity_mbs1_joint0" class="mbs1/" joint="mbs1/mbs1_joint0" kv="0.050000000000000003"/>
    <position name="mbs1/actuator_position_mbs1_link0_joint1" class="mbs1/" joint="mbs1/mbs1_link0_joint1" kp="5"/>
    <velocity name="mbs1/actuator_velocity_mbs1_link0_joint1" class="mbs1/" joint="mbs1/mbs1_link0_joint1" kv="0.050000000000000003"/>
    <position name="mbs1/actuator_position_mbs1_link0_link1_joint1" class="mbs1/" joint="mbs1/mbs1_link0_link1_joint1" kp="5"/>
    <velocity name="mbs1/actuator_velocity_mbs1_link0_link1_joint1" class="mbs1/" joint="mbs1/mbs1_link0_link1_joint1" kv="0.050000000000000003"/>
    <position name="mbs1/actuator_position_mbs1_link0_link1_joint2" class="mbs1/" joint="mbs1/mbs1_link0_link1_joint2" kp="5"/>
    <velocity name="mbs1/actuator_velocity_mbs1_link0_link1_joint2" class="mbs1/" joint="mbs1/mbs1_link0_link1_joint2" kv="0.050000000000000003"/>
    <position name="mbs1/actuator_position_mbs1_joint1" class="mbs1/" joint="mbs1/mbs1_joint1" kp="5"/>
    <velocity name="mbs1/actuator_velocity_mbs1_joint1" class="mbs1/" joint="mbs1/mbs1_joint1" kv="0.050000000000000003"/>
    <position name="mbs1/actuator_position_mbs1_joint2" class="mbs1/" joint="mbs1/mbs1_joint2" kp="5"/>
    <velocity name="mbs1/actuator_velocity_mbs1_joint2" class="mbs1/" joint="mbs1/mbs1_joint2" kv="0.050000000000000003"/>
  </actuator>
</mujoco>

"""