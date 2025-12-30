import os
import re
import subprocess

# MuJoCo compiler settings to be inserted into URDF
MUJOCO_INSERT = """
  <mujoco>
    <compiler angle="radian" meshdir="meshes/" balanceinertia="true" discardvisual="false"/>
  </mujoco>
"""

# Option settings to be inserted after XML generation
XML_INSERT_OPTION = """
    <option timestep="0.002" integrator="implicit" density="1.225" viscosity="1.8e-5" cone="elliptic" impratio="10" />
"""

# Asset definitions including meshes and materials
XML_INSERT_ASSET = """
        <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
        <material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance="0.2" specular="1.0" shininess="0.5"/>
"""

# World body elements including floor and lighting
XML_INSERT_WORLD = """
        <light directional="true" diffuse=".8 .8 .8" specular=".5 .5 .5" pos="0 0 5" dir="0 0 -1" castshadow="true"/>
        <geom name="floor" type="plane" pos="0 0 0" size="0 0 0.1" contype="1" conaffinity="1" material="grid"/>
"""

# Actuator and sensor configurations to be inserted after worldbody
XML_INSERT_AFTER_WORLD = """
    <actuator>
        <motor name="rotor_joint_1" joint="rotor_joint_1" />
        <motor name="rotor_joint_2" joint="rotor_joint_2" />
        <motor name="rotor_joint_3" joint="rotor_joint_3" />
        <motor name="rotor_joint_4" joint="rotor_joint_4" />
        <motor name="rotor_joint_thrust1" site="rotor_joint_thrust1" gear="0 0 1 0 0 -0.016" />
        <motor name="rotor_joint_thrust2" site="rotor_joint_thrust2" gear="0 0 1 0 0 0.016" />
        <motor name="rotor_joint_thrust3" site="rotor_joint_thrust3" gear="0 0 1 0 0 -0.016" />
        <motor name="rotor_joint_thrust4" site="rotor_joint_thrust4" gear="0 0 1 0 0 0.016" />
        <position name="joint_1" joint="joint_1" kp="748.6" kv="0.547" forcerange="-4.905 4.905" ctrlrange="-2.6485 2.6485" />
        <position name="joint_2" joint="joint_2" kp="524.0" kv="0.727" forcerange="-3.43 3.43" ctrlrange="0 3.4907" />
        <position name="joint_3" joint="joint_3" kp="524.0" kv="0.727" forcerange="-3.43 3.43" ctrlrange="-2.6485 2.6485" />
        <position name="joint_4" joint="joint_4" kp="524.0" kv="0.727" forcerange="-3.43 3.43" ctrlrange="-3.1416 3.1416" />
        <position name="joint_5" joint="joint_5" kp="212.6" kv="0.133" forcerange="-1.3916 1.3916" ctrlrange="-1.723 0" />
        <position name="joint_gripper_left" joint="joint_gripper_left" kp="2000.0" kv="124.0" forcerange="-49.06 49.06" ctrlrange="0 0.04225" />
        <position name="joint_gripper_right" joint="joint_gripper_right" kp="2000.0" kv="124.0" forcerange="-49.06 49.06" ctrlrange="-0.04225 0"/>
    </actuator>
    <sensor>
        <framepos name="framepos" objtype="site" objname="base_link_origin" />
        <framequat name="framequat" objtype="site" objname="base_link_origin" />
        <velocimeter name="velocimeter" site="base_link_origin" />
        <gyro name="gyro" site="base_link_origin" />
    </sensor>
"""


def preprocess_urdf(urdf_path: str) -> str:
    """
    Preprocess URDF file by adding MuJoCo-specific compiler settings.

    Args:
        urdf_path (str): Path to the input URDF file

    Returns:
        str: Path to the temporary preprocessed URDF file
    """
    with open(urdf_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Fix mesh file paths to use relative paths
    content = re.sub(r'filename="[^"]*meshes/', 'filename="meshes/', content)

    urdf_dir = os.path.dirname(urdf_path)
    urdf_name = os.path.splitext(os.path.basename(urdf_path))[0]
    tmp_path = os.path.join(urdf_dir, urdf_name + "_tmp.urdf")

    # Skip preprocessing if MuJoCo tags already exist
    if "<mujoco" in content:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(content)
        return tmp_path

    # Insert MuJoCo compiler settings after robot tag
    content = re.sub(r"(<robot[^>]*>)", r"\1" + MUJOCO_INSERT, content)

    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(content)

    return tmp_path


def postprocess_xml(xml_path: str):
    """
    Postprocess the generated XML file to add custom configurations and clean up formatting.

    Args:
        xml_path (str): Path to the XML file to be processed
    """
    with open(xml_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Remove duplicate geom elements (keep only the first one)
    lines = content.splitlines()
    cleaned_lines = []
    skip_next_geom = False

    for i, line in enumerate(lines):
        # Skip empty lines
        if not line.strip():
            continue

        # Remove duplicate </worldbody> tags
        if line.strip() == "</worldbody>" and i < len(lines) - 1 and lines[i + 1].strip() == "</worldbody>":
            continue

        # Remove duplicate geom elements
        if '<geom type="mesh"' in line and not skip_next_geom:
            cleaned_lines.append(line)
            # Check if next line is also a geom with the same mesh
            if i < len(lines) - 1 and '<geom type="mesh"' in lines[i + 1]:
                # Extract mesh name from current line
                mesh_match = re.search(r'mesh="([^"]*)"', line)
                if mesh_match:
                    mesh_name = mesh_match.group(1)
                    # Check if next line has the same mesh
                    if mesh_name in lines[i + 1]:
                        skip_next_geom = True
                        continue
            skip_next_geom = False
        elif skip_next_geom and '<geom type="mesh"' in line:
            skip_next_geom = False
            continue
        else:
            cleaned_lines.append(line)
            skip_next_geom = False

    # Join lines back into content
    content = "\n".join(cleaned_lines)

    # Insert option settings after compiler tag
    if "<compiler" in content and "<option" not in content:
        # Find the compiler tag and insert option after it
        compiler_pattern = r"(<compiler[^>]*/>)"
        replacement = r"\1\n" + XML_INSERT_OPTION
        content = re.sub(compiler_pattern, replacement, content)

    # Insert asset definitions
    if "<asset>" in content and 'texture name="grid"' not in content:
        content = content.replace("<asset>", "<asset>\n" + XML_INSERT_ASSET)

    # Insert world body elements
    if "<worldbody>" in content and 'geom name="floor"' not in content:
        # Insert after <worldbody> tag
        content = content.replace("<worldbody>", "<worldbody>\n" + XML_INSERT_WORLD)

    # Insert actuator and sensor configurations
    if "</worldbody>" in content and "<actuator>" not in content:
        content = content.replace("</worldbody>", "</worldbody>\n" + XML_INSERT_AFTER_WORLD)

    # Fix collision properties to enable collisions
    # content = content.replace('contype="0"', 'contype="1"')
    # content = content.replace('conaffinity="0"', 'conaffinity="1"')

    # Add base_link_origin site if it doesn't exist
    if 'site name="base_link_origin"' not in content:
        # Find a good place to insert the site, e.g., after the base_link geom
        base_link_geom_pos = content.find('<geom type="mesh"')
        if base_link_geom_pos != -1:
            # Find the end of the geom tag
            geom_end = content.find(">", base_link_geom_pos)
            if geom_end != -1:
                site_insert = (
                    '\n        <site name="base_link_origin" type="sphere" size="0.01" rgba="1 0 0 0.5" pos="0 0 0"/>'
                )
                content = content[: geom_end + 1] + site_insert + content[geom_end + 1 :]

    # Add thrust sites for rotors
    rotor_sites = {
        "rotor_1": "rotor_joint_thrust1",
        "rotor_2": "rotor_joint_thrust2",
        "rotor_3": "rotor_joint_thrust3",
        "rotor_4": "rotor_joint_thrust4",
    }

    for rotor_name, site_name in rotor_sites.items():
        if f'site name="{site_name}"' not in content:
            # Find the rotor body
            rotor_pattern = f'<body name="{rotor_name}"[^>]*>.*?</body>'
            rotor_match = re.search(rotor_pattern, content, re.DOTALL)
            if rotor_match:
                rotor_content = rotor_match.group(0)
                # Insert site before the closing body tag
                site_insert = f'\n                <site name="{site_name}" type="cylinder" size="0.01 0.005" pos="0 0 0" rgba="1 0 0 0.5"/>'
                new_rotor_content = rotor_content.replace("</body>", site_insert + "\n        </body>")
                content = content.replace(rotor_content, new_rotor_content)

    # 移除所有空行
    lines = content.splitlines()
    non_empty_lines = [line.rstrip() for line in lines if line.strip()]
    content = "\n".join(non_empty_lines)

    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(content)


def compile_urdf(urdf_path: str, xml_path: str):
    """
    Compile URDF file to MuJoCo XML format with custom configurations.

    Args:
        urdf_path (str): Path to input URDF file
        xml_path (str): Path to output XML file
    """
    # Update this path to match your MuJoCo installation
    mujoco_bin = os.path.expanduser("C://Users/XXY/.mujoco/mujoco210/bin/compile.exe")

    if not os.path.isfile(mujoco_bin):
        raise FileNotFoundError(f"MuJoCo compiler not found at {mujoco_bin}")

    # Remove existing XML file if it exists
    if os.path.exists(xml_path):
        os.remove(xml_path)

    # Preprocess URDF and compile to XML
    preprocessed_urdf_path = preprocess_urdf(urdf_path)
    cmd = [mujoco_bin, preprocessed_urdf_path, xml_path]

    try:
        subprocess.run(cmd, check=True)
        # Apply post-processing to the generated XML
        postprocess_xml(xml_path)
        print(f"Conversion successful: {urdf_path} -> {xml_path}")

        # Clean up temporary file
        if os.path.exists(preprocessed_urdf_path):
            os.remove(preprocessed_urdf_path)

    except subprocess.CalledProcessError as e:
        print(f"Conversion failed with error: {e}")
        # Clean up temporary file on failure
        if os.path.exists(preprocessed_urdf_path):
            os.remove(preprocessed_urdf_path)


if __name__ == "__main__":
    """
    Main execution block for URDF to MuJoCo XML conversion.
    """
    target_name = "x500_arm_v2"

    urdf_path = f"../station/flying_hand/description/{target_name}/{target_name}.urdf"
    xml_path = f"../station/flying_hand/description/{target_name}/{target_name}.xml"

    compile_urdf(urdf_path, xml_path)
