import os
import re
import subprocess

MUJOCO_INSERT = """
  <mujoco>
    <compiler meshdir="meshes/" balanceinertia="true" discardvisual="false"/>
  </mujoco>
"""

XML_INSERT_WORLD = """
      <geom name="floor" pos="0 0 0" size="0 0 0.1" type="plane"/>
        <light directional="true" diffuse=".2 .2 .2" specular="0 0 0" pos="0 0 5" dir="0 0 -1"/>
        <light mode="targetbodycom" directional="false" diffuse=".8 .8 .8" specular="0.3 0.3 0.3" pos="0 0 4.0" dir="0 0 -1"/>
"""

XML_INSERT_AFTER_WORLD = """
    <actuator>
      <velocity name="rotor1" joint="rotor_joint_1" gear="1" />
      <velocity name="rotor2" joint="rotor_joint_2" gear="1" />
      <velocity name="rotor3" joint="rotor_joint_3" gear="1" />
      <velocity name="rotor4" joint="rotor_joint_4" gear="1" />
      <position name="motor1" joint="joint_1" gear="1" kp="50" dampratio="1" />
      <position name="motor2" joint="joint_2" gear="1" kp="50" dampratio="1" />
      <position name="motor3" joint="joint_3" gear="1" kp="50" dampratio="1" />
      <position name="motor4" joint="joint_4" gear="1" kp="1" dampratio="1"/>
      <position name="motor5" joint="joint_5" gear="1" kp="1" dampratio="1" />
    </actuator>
"""


def preprocess_urdf(urdf_path: str) -> str:
    with open(urdf_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    content = "".join(lines)
    content = re.sub(r'filename="[^"]*meshes/', 'filename="meshes/', content)

    urdf_dir = os.path.dirname(urdf_path)
    urdf_name = os.path.splitext(os.path.basename(urdf_path))[0]
    tmp_path = os.path.join(urdf_dir, urdf_name + "_tmp.urdf")

    if "<mujoco" in content:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(content)
        return tmp_path

    new_lines = []
    buffer = ""
    inserted = False

    for line in content.splitlines(keepends=True):
        buffer += line
        new_lines.append(line)
        if not inserted:
            match = re.search(r"<robot\b[^>]*>", buffer)
            if match:
                new_lines.append(MUJOCO_INSERT + "\n")
                inserted = True
                buffer = ""

    with open(tmp_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    return tmp_path


def postprocess_xml(xml_path: str):
    with open(xml_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    inserted_world = False

    for line in lines:
        if not inserted_world and "<worldbody" in line:
            new_lines.append(line.rstrip("\n"))
            new_lines.append(XML_INSERT_WORLD)
            inserted_world = True
        elif "</worldbody>" in line:
            new_lines.append(line.rstrip("\n"))
            new_lines.append(XML_INSERT_AFTER_WORLD)
        elif 'contype="0"' in line:
            line = line.replace('contype="0"', 'contype="1"')
            line = line.replace('conaffinity="0"', 'conaffinity="1"')
            new_lines.append(line)
        else:
            new_lines.append(line)

    with open(xml_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


def compile_urdf(urdf_path: str, xml_path: str):
    mujoco_bin = os.path.expanduser("C://Users/XXY/.mujoco/mujoco210/bin/compile.exe")

    if not os.path.isfile(mujoco_bin):
        raise FileNotFoundError(f"MuJoCo compiler not found at {mujoco_bin}")

    if os.path.exists(xml_path):
        os.remove(xml_path)

    preprocessed_urdf_path = preprocess_urdf(urdf_path)
    cmd = [mujoco_bin, preprocessed_urdf_path, xml_path]

    try:
        subprocess.run(cmd, check=True)
        postprocess_xml(xml_path)
        print(f"转换成功: {urdf_path} -> {xml_path}")
    except subprocess.CalledProcessError as e:
        print(f"转换失败，错误信息: {e}")


if __name__ == "__main__":
    target_name = "x500_arm"
    urdf_path = os.path.expanduser(f"G://NEU_Tele/neutele/station/flying_hand/urdf/x500_arm/{target_name}.urdf")
    xml_path = os.path.expanduser(f"G://NEU_Tele/neutele/station/flying_hand/urdf/x500_arm/{target_name}.xml")
    compile_urdf(urdf_path, xml_path)
