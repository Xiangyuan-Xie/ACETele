import argparse
import copy
import os
import platform
import re
import shutil
import subprocess
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

import coacd
import pinocchio as pin
import trimesh

from acetele.utils.math import quat_mul, quat_rotate


class URDF2MJCFConverter:
    """Convert URDF to MuJoCo MJCF, inject common options/assets/sensors, and post-process."""

    # Constants
    MUJOCO_COMPILER_TAG = textwrap.dedent("""
        <mujoco>
            <compiler
                angle="radian"
                meshdir="meshes/"
                texturedir="meshes/"
                balanceinertia="true"
                discardvisual="false"
            />
        </mujoco>
    """).strip()

    XML_OPTION_TAG = textwrap.dedent("""
        <option
            density="1.225"
            timestep="0.002"
            integrator="implicit"
            viscosity="1.8e-5"
            cone="elliptic"
            impratio="10"
            magnetic="2.73e-5 0 -4.54e-5"
        />
    """).strip()

    ATTRIB_ORDER = [
        "name",
        "class",
        "type",
        "mesh",
        "material",
        "size",
        "pos",
        "quat",
        "axis",
        "fromto",
        "mass",
        "density",
        "rgba",
        "group",
        "contype",
        "conaffinity",
        "condim",
        "kp",
        "kv",
        "gear",
        "joint",
        "site",
        "objtype",
        "objname",
        "forcerange",
        "ctrlrange",
        "ctrllimited",
        "forcelimited",
    ]

    XML_ASSET_TAG = textwrap.dedent("""
        <texture name="grid" type="2d" builtin="checker" width="512" height="512"
                 rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
        <material name="grid" texture="grid" texrepeat="1 1" texuniform="true"
                 reflectance="0.2" specular="1.0" shininess="0.5"/>
    """).strip()

    XML_WORLD_TAG = textwrap.dedent("""
        <light directional="true" diffuse=".8 .8 .8" specular=".5 .5 .5" pos="0 0 5" dir="0 0 -1" castshadow="true"/>
        <geom name="floor" type="plane" pos="0 0 0" size="0 0 0.1" material="grid"/>
    """).strip()

    XML_ACTUATORS_SENSORS = textwrap.dedent("""
        <actuator>
            <position
                name="joint_1" joint="joint_1" kp="748.6" kv="0.547"
                forcerange="-4.905 4.905" ctrlrange="-2.6485 2.6485"
            />
            <position
                name="joint_2" joint="joint_2" kp="524.0" kv="0.727"
                forcerange="-3.43 3.43" ctrlrange="0 3.4907"
            />
            <position
                name="joint_3" joint="joint_3" kp="524.0" kv="0.727"
                forcerange="-3.43 3.43" ctrlrange="-2.6485 2.6485"
            />
            <position
                name="joint_4" joint="joint_4" kp="524.0" kv="0.727"
                forcerange="-3.43 3.43" ctrlrange="-3.1416 3.1416"
            />
            <position
                name="joint_5" joint="joint_5" kp="212.6" kv="0.133"
                forcerange="-1.3916 1.3916" ctrlrange="-1.723 0"
            />
            <position
                name="joint_gripper_left" joint="joint_gripper_left" kp="2000.0" kv="124.0"
                forcerange="-49.06 49.06" ctrlrange="-0.04225 0"
            />
            <position
                name="joint_gripper_right" joint="joint_gripper_right" kp="2000.0" kv="124.0"
                forcerange="-49.06 49.06" ctrlrange="0 0.04225"
            />
        </actuator>
        <sensor>
            <framepos name="framepos" objtype="site" objname="base_link_origin" />
            <framequat name="framequat" objtype="site" objname="base_link_origin" />
            <framelinvel name="framelinvel" objtype="site" objname="base_link_origin" />
            <gyro name="gyro" site="base_link_origin" />
            <accelerometer name="accelerometer" site="base_link_origin" />
            <magnetometer name="magnetometer" site="base_link_origin" />
        </sensor>
    """).strip()

    def __init__(
        self,
        target: str,
        floating: bool = False,
        decompose: bool = False,
        safety_margin: float = 0.05,
        q0: str = "",
        mujoco_bin: Optional[str] = None,
    ):
        self.target = target
        self.floating = floating
        self.decompose = decompose
        self.safety_margin = safety_margin
        self.q0_str = q0
        self.mujoco_bin = mujoco_bin

        # Path setup
        self.base_dir = Path(__file__).parent.parent / "simulation" / "description"
        self.urdf_path = self.base_dir / self.target / f"{self.target}.urdf"
        self.mesh_dir = self.urdf_path.parent / "meshes"
        self.xml_path = self.urdf_path.parent / f"{self.target}.xml"

        self.initial_q = self._parse_q0(q0)

    def _parse_q0(self, q0_str: str) -> Dict[str, float]:
        initial_q = {}
        if q0_str:
            for pair in q0_str.split(","):
                if "=" in pair:
                    k, v = pair.split("=")
                    try:
                        initial_q[k.strip()] = float(v)
                    except ValueError:
                        print(f"Invalid q0 value for {k}: {v}")
        return initial_q

    @staticmethod
    def indent_xml(elem: ET.Element, level: int = 0) -> None:
        """In-place prettyprint formatter for ElementTree."""
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for child in elem:
                URDF2MJCFConverter.indent_xml(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = i
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    @staticmethod
    def inject_xml(parent: ET.Element, xml_content: str, index: int = -1) -> None:
        """Parses and injects XML content into parent element at specified index."""
        try:
            fragment = ET.fromstring(f"<root>{xml_content}</root>")
            for child in list(fragment):
                if index < 0:
                    parent.append(child)
                else:
                    parent.insert(index, child)
                    index += 1
        except ET.ParseError:
            pass

    def clean_artifacts(self) -> None:
        """Removes existing XML and decomposed meshes to ensure fresh compilation."""
        if self.xml_path.exists():
            try:
                os.remove(self.xml_path)
                print(f"Removed existing output: {self.xml_path}")
            except OSError as e:
                print(f"Error removing {self.xml_path}: {e}")

        # Remove decomposed meshes
        if self.mesh_dir.exists():
            count = 0
            for f in self.mesh_dir.glob("*_decomp_*.stl"):
                try:
                    os.remove(f)
                    count += 1
                except OSError as e:
                    print(f"Error deleting {f}: {e}")
            if count > 0:
                print(f"Removed {count} decomposed mesh files.")

    def sort_attributes(self, elem: ET.Element) -> None:
        """Sorts attributes of an element and its children based on a provided order list."""
        if elem.attrib:
            sorted_attrib = {}
            # Add keys in order
            for key in self.ATTRIB_ORDER:
                if key in elem.attrib:
                    sorted_attrib[key] = elem.attrib[key]

            # Add remaining keys
            for key, value in elem.attrib.items():
                if key not in sorted_attrib:
                    sorted_attrib[key] = value

            # Reset and populate
            elem.attrib.clear()
            elem.attrib.update(sorted_attrib)

        for child in elem:
            self.sort_attributes(child)

    def add_collision_exclusions(self, root: ET.Element) -> None:
        """Adds collision exclusion between parent-child and sibling bodies."""
        contact = root.find("contact")
        if contact is None:
            contact = ET.Element("contact")
            root.append(contact)

        existing_excludes = set()
        for exclude in contact.findall("exclude"):
            b1 = exclude.get("body1")
            b2 = exclude.get("body2")
            if b1 and b2:
                existing_excludes.add(tuple(sorted((b1, b2))))

        def traverse(body):
            parent_name = body.get("name")
            children = [c for c in body if c.tag == "body"]

            # Parent-Child Exclusion
            for i, child in enumerate(children):
                child_name = child.get("name")
                if not parent_name or not child_name:
                    continue

                # Exclude parent-child
                pair = tuple(sorted((parent_name, child_name)))
                if pair not in existing_excludes and parent_name != "world":
                    exclude_elem = ET.SubElement(contact, "exclude")
                    exclude_elem.set("body1", parent_name)
                    exclude_elem.set("body2", child_name)
                    existing_excludes.add(pair)

                # Sibling-Sibling Exclusion
                for j in range(i):
                    sibling = children[j]
                    sibling_name = sibling.get("name")
                    if sibling_name:
                        pair_sib = tuple(sorted((child_name, sibling_name)))
                        if pair_sib not in existing_excludes:
                            exclude_elem = ET.SubElement(contact, "exclude")
                            exclude_elem.set("body1", child_name)
                            exclude_elem.set("body2", sibling_name)
                            existing_excludes.add(pair_sib)

                traverse(child)

        worldbody = root.find("worldbody")
        if worldbody is not None:
            for child in worldbody.findall("body"):
                traverse(child)

    def decompose_mesh(
        self, mesh_path: str, output_dir: str, threshold: float = 0.2, resolution: int = 50
    ) -> List[str]:
        """Decompose a single mesh into convex parts using CoACD."""
        if coacd is None or trimesh is None:
            raise ImportError("coacd and trimesh are required for convex decomposition.")

        print(f"Decomposing mesh: {os.path.basename(mesh_path)}")
        try:
            mesh = trimesh.load(mesh_path, force="mesh")
        except Exception as e:
            print(f"Error loading mesh {mesh_path}: {e}")
            return [mesh_path]

        coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)
        parts = coacd.run_coacd(coacd_mesh, threshold=threshold, preprocess_resolution=resolution)

        mesh_name = os.path.splitext(os.path.basename(mesh_path))[0]
        output_paths = []

        for idx, (v, f) in enumerate(parts):
            part_mesh = trimesh.Trimesh(v, f)
            export_name = f"{mesh_name}_decomp_{idx}.stl"
            export_path = os.path.join(output_dir, export_name)
            part_mesh.export(export_path)
            output_paths.append(export_path)

        return output_paths

    def process_urdf_collisions(
        self, urdf_path: str, mesh_dir: str, threshold: float = 0.2, resolution: int = 50
    ) -> str:
        """Parse URDF, decompose collision meshes, and return path to new temporary URDF."""
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        decomposed_count = 0

        for link in root.findall("link"):
            collisions = link.findall("collision")
            for col in collisions:
                geometry = col.find("geometry")
                if geometry is None:
                    continue
                mesh_elem = geometry.find("mesh")
                if mesh_elem is None:
                    continue

                filename = mesh_elem.get("filename")
                if not filename:
                    continue

                clean_name = os.path.basename(filename)
                mesh_path = os.path.join(mesh_dir, clean_name)

                if not os.path.exists(mesh_path):
                    print(f"Warning: Mesh file not found: {mesh_path}. Skipping.")
                    continue

                parts_paths = self.decompose_mesh(mesh_path, mesh_dir, threshold, resolution)

                if len(parts_paths) <= 1 and parts_paths[0] == mesh_path:
                    continue

                link.remove(col)
                for part_path in parts_paths:
                    new_col = copy.deepcopy(col)
                    new_geom = new_col.find("geometry")
                    if new_geom is None:
                        continue
                    new_mesh = new_geom.find("mesh")
                    if new_mesh is None:
                        continue

                    part_name = os.path.basename(part_path)
                    if "/" in filename or "\\" in filename:
                        prefix = os.path.dirname(filename)
                        new_filename = os.path.join(prefix, part_name).replace("\\", "/")
                    else:
                        new_filename = part_name

                    new_mesh.set("filename", new_filename)
                    link.append(new_col)
                    decomposed_count += 1

        print(f"Decomposition complete. Replaced collisions with {decomposed_count} convex parts.")

        urdf_dir = os.path.dirname(urdf_path)
        urdf_name = os.path.splitext(os.path.basename(urdf_path))[0]
        new_urdf_path = os.path.join(urdf_dir, f"{urdf_name}_decomposed_tmp.urdf")
        tree.write(new_urdf_path, encoding="utf-8", xml_declaration=True)
        return new_urdf_path

    def calculate_min_z(self, urdf_path: str) -> float:
        """Calculate the lowest Z coordinate of the robot using Pinocchio and Trimesh."""
        if pin is None:
            print("Warning: Pinocchio not installed. Cannot calculate height.")
            return 0.0

        try:
            urdf_dir = os.path.dirname(urdf_path)
            package_dir = os.path.dirname(urdf_dir)
            model, collision_model, visual_model = pin.buildModelsFromUrdf(
                urdf_path, package_dirs=package_dir, root_joint=pin.JointModelFreeFlyer()
            )
            data = model.createData()
            collision_data = collision_model.createData()

            q = pin.neutral(model)

            if self.initial_q:
                for name, val in self.initial_q.items():
                    if model.existJointName(name):
                        joint_id = model.getJointId(name)
                        idx_q = model.joints[joint_id].idx_q
                        nq = model.joints[joint_id].nq
                        if nq == 1:
                            q[idx_q] = val

            pin.forwardKinematics(model, data, q)
            pin.updateGeometryPlacements(model, data, collision_model, collision_data)

            min_z = float("inf")
            found_geom = False

            for i, geom_obj in enumerate(collision_model.geometryObjects):
                placement = collision_data.oMg[i]
                transform = placement.homogeneous

                if trimesh is not None and geom_obj.meshPath and os.path.exists(geom_obj.meshPath):
                    try:
                        mesh = trimesh.load(geom_obj.meshPath, force="mesh")
                        mesh.apply_transform(transform)
                        mesh_min_z = mesh.bounds[0][2]
                        if mesh_min_z < min_z:
                            min_z = mesh_min_z
                            found_geom = True
                        continue
                    except Exception as e:
                        print(f"Warning: Failed to load mesh for height calc: {e}")

                pos = placement.translation
                if pos[2] < min_z:
                    min_z = pos[2]
                    found_geom = True

            if not found_geom:
                return 0.0

            return min_z

        except Exception as e:
            print(f"Error calculating height: {e}")
            return 0.0

    def preprocess_urdf(self, urdf_path: str, height_offset: float = 0.0) -> str:
        """Prepare URDF for compilation."""
        with open(urdf_path, "r", encoding="utf-8") as f:
            content = f.read()

        content = re.sub(r'filename="[^"]*meshes/', 'filename="meshes/', content)

        if self.floating:
            links = set(re.findall(r'<link\s+name="([^"]+)"', content))
            children = set(re.findall(r'<child\s+link="([^"]+)"', content))
            roots = list(links - children)

            if len(roots) == 1:
                root_link = roots[0]
                print(f"Injecting floating joint for root: {root_link} with offset {height_offset:.4f}m")

                floating_insert = f"""
  <link name="world_root_dummy_link"/>
  <joint name="floating_base_joint" type="floating">
    <origin xyz="0 0 {height_offset}" rpy="0 0 0"/>
    <parent link="world_root_dummy_link"/>
    <child link="{root_link}"/>
  </joint>
"""
                content = re.sub(r"(<robot[^>]*>)", r"\1" + floating_insert, content)
            else:
                print("Warning: Could not identify unique root link. Skipping floating joint injection.")

        if "<mujoco" not in content:
            content = re.sub(r"(<robot[^>]*>)", r"\1" + self.MUJOCO_COMPILER_TAG, content)

        urdf_dir = os.path.dirname(urdf_path)
        urdf_name = os.path.splitext(os.path.basename(urdf_path))[0]
        tmp_path = os.path.join(urdf_dir, f"{urdf_name}_tmp.urdf")

        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(content)

        return tmp_path

    def postprocess_xml(self, xml_path: str, height_offset: float = 0.0) -> None:
        """Modify the generated MJCF XML."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        worldbody = root.find("worldbody")
        body_parent: Dict[str, Optional[str]] = {}
        body_elem: Dict[str, ET.Element] = {}
        body_transform: Dict[str, Dict[str, List[float]]] = {}

        # Build body parent-child map and local transforms
        def walk_bodies(elem: ET.Element, parent_name: Optional[str] = None):
            for body in elem.findall("body"):
                name = body.get("name")
                if not name:
                    continue
                body_parent[name] = parent_name
                body_elem[name] = body
                # Parse local pos/quat
                pos_str = body.get("pos", "0 0 0")
                quat_str = body.get("quat", "1 0 0 0")
                try:
                    pos = [float(x) for x in pos_str.split()]
                except Exception:
                    pos = [0.0, 0.0, 0.0]
                try:
                    quat = [float(x) for x in quat_str.split()]
                except Exception:
                    quat = [1.0, 0.0, 0.0, 0.0]
                body_transform[name] = {"pos": pos, "quat": quat}
                walk_bodies(body, name)

        if worldbody is not None:
            walk_bodies(worldbody, None)

        def compose_to_base(child_name: str, base_name: str = "base_link"):
            # Compose transform from base to child: p = R_base * offset + p_base, return offset in base frame
            path = []
            cur: Optional[str] = child_name
            while cur is not None and cur in body_parent:
                path.append(cur)
                cur = body_parent[cur]
                if cur == base_name:
                    path.append(cur)
                    break
            if not path or path[-1] != base_name:
                return None
            # Accumulate world transform along path
            acc_quat = [1.0, 0.0, 0.0, 0.0]
            acc_pos = [0.0, 0.0, 0.0]
            for i in range(len(path) - 1, 0, -1):
                node = path[i - 1]
                tf = body_transform.get(node, {"pos": [0.0, 0.0, 0.0], "quat": [1, 0, 0, 0]})
                # acc_pos = acc_pos + acc_quat * tf.pos
                rotated = quat_rotate(acc_quat, tf["pos"])
                acc_pos = [acc_pos[0] + rotated[0], acc_pos[1] + rotated[1], acc_pos[2] + rotated[2]]
                # acc_quat = acc_quat ∘ tf.quat
                acc_quat = quat_mul(acc_quat, tf["quat"])
            # base rotation
            base_quat = body_transform.get(base_name, {"pos": [0, 0, 0], "quat": [1, 0, 0, 0]})["quat"]
            # offset in base frame: R_base^T * (acc_pos)
            # For unit quaternion, inverse = conjugate
            inv_base = [base_quat[0], -base_quat[1], -base_quat[2], -base_quat[3]]
            offset = quat_rotate(inv_base, acc_pos)
            return offset

        # 1. Clean up worldbody (remove duplicate meshes)
        if worldbody is not None:
            seen = set()
            for geom in list(worldbody.findall("geom")):
                if geom.get("type") == "mesh":
                    name = geom.get("mesh")
                    if name in seen:
                        worldbody.remove(geom)
                    else:
                        seen.add(name)

        # 2. Insert World Elements
        if worldbody is not None and not any(g.get("name") == "floor" for g in worldbody.findall("geom")):
            self.inject_xml(worldbody, self.XML_WORLD_TAG, 0)

        # 3. Insert Compiler Options
        if root.find("option") is None:
            compiler = root.find("compiler")
            idx = list(root).index(compiler) if compiler is not None else 0
            self.inject_xml(root, self.XML_OPTION_TAG, idx + 1)

        # 4. Insert Assets
        asset = root.find("asset")
        if asset is None:
            asset = ET.Element("asset")
            root.insert(0, asset)
        if not any(t.get("name") == "grid" for t in asset.findall("texture")):
            self.inject_xml(asset, self.XML_ASSET_TAG)

        # 5. Insert Actuators & Sensors
        if root.find("actuator") is None:
            self.inject_xml(root, self.XML_ACTUATORS_SENSORS)

        # 6. Add Sites
        if worldbody is not None:
            base_body = None
            for body in root.iter("body"):
                if body.get("name") == "base_link":
                    base_body = body
                    break

            if base_body is not None and not any(
                s.get("name") == "base_link_origin" for s in base_body.findall("site")
            ):
                base_body.append(
                    ET.Element(
                        "site",
                        {
                            "name": "base_link_origin",
                            "type": "sphere",
                            "size": "0.01",
                            "rgba": "1 0 0 0.5",
                            "pos": "0 0 0",
                        },
                    )
                )

            # Insert rotor offset sites under base_link from existing rotor bodies if present
            rotor_names = [f"rotor_{i}" for i in range(1, 5)]
            offsets = {}
            for rn in rotor_names:
                if rn in body_elem:
                    off = compose_to_base(rn, "base_link")
                    if off is not None:
                        offsets[rn] = off
            # Fallback: look for any bodies containing 'rotor' substring
            if not offsets:
                for name in body_elem.keys():
                    if "rotor" in name and name != "base_link":
                        off = compose_to_base(name, "base_link")
                        if off is not None:
                            offsets[name] = off
            # Write sites if offsets found
            if base_body is not None:
                idx = 1
                for k, vec in list(offsets.items())[:4]:
                    pos_str = f"{vec[0]:.6f} {vec[1]:.6f} {vec[2]:.6f}"
                    site_name = f"rotor_offset_{idx}"
                    if not any(s.get("name") == site_name for s in base_body.findall("site")):
                        base_body.append(
                            ET.Element(
                                "site",
                                {
                                    "name": site_name,
                                    "type": "sphere",
                                    "size": "0.005",
                                    "rgba": "0 0 1 0.5",
                                    "pos": pos_str,
                                },
                            )
                        )
                    idx += 1

            rotor_sites = {f"rotor_{i}": f"rotor_joint_thrust{i}" for i in range(1, 5)}
            for body in root.iter("body"):
                name = body.get("name")
                if name in rotor_sites:
                    site_name = rotor_sites[name]
                    if not any(s.get("name") == site_name for s in body.findall("site")):
                        body.append(
                            ET.Element(
                                "site",
                                {
                                    "name": site_name,
                                    "type": "cylinder",
                                    "size": "0.01 0.005",
                                    "pos": "0 0 0",
                                    "rgba": "1 0 0 0.5",
                                },
                            )
                        )

        # 6.1 Add or convert rotor visuals to mocap and disable rotor dynamics
        if worldbody is not None:
            rotor_mesh_map: Dict[str, str] = {}
            for geom in root.iter("geom"):
                mesh = geom.get("mesh")
                if mesh and mesh.startswith("rotor_"):
                    m = re.match(r"(rotor_([1-4]))", mesh)
                    if m:
                        rotor_id = m.group(2)
                        rotor_mesh_map[rotor_id] = mesh

            # Prefer non-decomposed visual meshes from <asset> section if available
            asset = root.find("asset")
            asset_rotor_visual: Dict[str, str] = {}
            if asset is not None:
                for mesh_elem in asset.findall("mesh"):
                    name = mesh_elem.get("name", "")
                    match = re.match(r"rotor_([1-4])$", name)
                    if match:
                        asset_rotor_visual[match.group(1)] = name

            for rid, fallback_mesh in list(rotor_mesh_map.items())[:4]:
                mesh_name = asset_rotor_visual.get(rid) or fallback_mesh
                vis_name = f"rotor_{rid}_vis"
                existing = None
                for b in worldbody.findall("body"):
                    if b.get("name") == vis_name:
                        existing = b
                        break
                if existing is None:
                    nb = ET.Element("body", {"name": vis_name, "mocap": "true"})
                    ng = ET.Element(
                        "geom",
                        {
                            "type": "mesh",
                            "mesh": mesh_name,
                            "rgba": "1 1 1 1",
                            "group": "1",
                            "contype": "0",
                            "conaffinity": "0",
                        },
                    )
                    nb.append(ng)
                    worldbody.append(nb)
                else:
                    # Ensure existing visual uses the preferred mesh and non-collision settings
                    for g in existing.findall("geom"):
                        g.set("type", "mesh")
                        g.set("mesh", mesh_name)
                        g.set("group", "1")
                        g.set("contype", "0")
                        g.set("conaffinity", "0")
                    existing.set("mocap", "true")
            for body in root.iter("body"):
                name = body.get("name", "")
                if name.startswith("rotor_") and name != "base_link" and re.match(r"rotor_[1-4]$", name):
                    for j in list(body.findall("joint")):
                        body.remove(j)
                    for inert in list(body.findall("inertial")):
                        body.remove(inert)
                    for g in list(body.findall("geom")):
                        body.remove(g)

        # 7. Generate Keyframe
        qpos_values = []
        for joint in root.iter("joint"):
            j_name = joint.get("name")
            j_type = joint.get("type", "hinge")

            val = "0"
            if self.initial_q and j_name in self.initial_q:
                val = str(self.initial_q[j_name])

            if j_type == "free" or j_name == "floating_base_joint":
                qpos_values.extend(["0", "0", str(height_offset), "1", "0", "0", "0"])
            elif j_type == "ball":
                qpos_values.extend(["0", "0", "0", "0"])
            elif j_type == "slide":
                qpos_values.append(val)
            else:
                qpos_values.append(val)

        keyframe_str = " ".join(qpos_values)
        key = root.find("keyframe")
        if key is None:
            key = ET.Element("keyframe")
            root.append(key)

        for key_elem in key.findall("key"):
            if key_elem.get("name") == "home":
                key.remove(key_elem)

        key.append(ET.Element("key", name="home", qpos=keyframe_str))

        # 8. Add Missing Actuators
        actuator_elem = root.find("actuator")
        if actuator_elem is not None:
            existing_joints = {child.get("joint") for child in actuator_elem if child.get("joint")}

            for joint in root.iter("joint"):
                j_name = joint.get("name", "")
                j_type = joint.get("type", "hinge")
                if j_type != "free" and j_name not in existing_joints and j_name != "floating_base_joint":
                    limited = joint.get("limited", "false") == "true"
                    r = joint.get("range", "0 0")
                    ctrl_range = r if limited else "-3.14 3.14"

                    is_gripper = "gripper" in (j_name or "")
                    kp = "2000" if is_gripper else "500"
                    kv = "124" if is_gripper else "50"

                    new_act = ET.Element(
                        "position",
                        {
                            "name": j_name,
                            "joint": j_name,
                            "kp": kp,
                            "kv": kv,
                            "ctrllimited": "true",
                            "ctrlrange": ctrl_range,
                            "forcelimited": "true",
                            "forcerange": "-100 100",
                        },
                    )
                    actuator_elem.append(new_act)
                    print(f"Added default actuator for: {j_name}")
            for child in list(actuator_elem):
                jn = child.get("joint", "")
                if jn and jn.startswith("rotor_joint_"):
                    actuator_elem.remove(child)

        # 9. Granular Geom Settings
        for geom in root.iter("geom"):
            if geom.get("name") == "floor":
                geom.set("group", "0")
                geom.set("contype", "1")
                geom.set("conaffinity", "1")
                continue

            mesh_name = geom.get("mesh", "")
            is_collision = False
            if "_decomp_" in mesh_name:
                is_collision = True
            elif geom.get("group") == "0" or geom.get("group") == "3":
                is_collision = True
            elif geom.get("contype") == "1" or geom.get("conaffinity") == "1":
                if geom.get("group") != "1":
                    is_collision = True

            if mesh_name.startswith("rotor_"):
                is_collision = False
            if is_collision:
                geom.set("group", "3")
                geom.set("contype", "1")
                geom.set("conaffinity", "1")
            else:
                geom.set("group", "1")
                geom.set("contype", "0")
                geom.set("conaffinity", "0")

        # 10. Add Collision Exclusions
        self.add_collision_exclusions(root)

        # 11. Sort Attributes
        self.sort_attributes(root)

        self.indent_xml(root)
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)

    def _find_mujoco_binary(self) -> str:
        """Finds the MuJoCo 'compile' binary using default paths or PATH."""
        if self.mujoco_bin:
            if os.path.exists(self.mujoco_bin):
                return self.mujoco_bin
            print(f"Warning: Specified binary not found at {self.mujoco_bin}. Searching defaults...")

        # Check PATH
        bin_path = shutil.which("compile")
        if bin_path:
            return bin_path

        # Check platform-specific default paths
        system = platform.system()
        candidates = []

        if system == "Windows":
            # Check standard Program Files locations
            program_files = [
                os.environ.get("PROGRAMFILES", r"C:\Program Files"),
                os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"),
            ]

            for pf in program_files:
                if pf:
                    candidates.append(os.path.join(pf, "MuJoCo", "bin", "compile.exe"))
                    candidates.append(os.path.join(pf, "mujoco", "bin", "compile.exe"))

            # Check user home directory
            user_profile = os.environ.get("USERPROFILE")
            if user_profile:
                candidates.append(os.path.join(user_profile, ".mujoco", "bin", "compile.exe"))
                candidates.append(os.path.join(user_profile, "mujoco", "bin", "compile.exe"))

            # Legacy/Simple paths
            candidates.append(r"C:\mujoco\bin\compile.exe")

        elif system == "Linux":
            home = os.environ.get("HOME", "/root")
            candidates.append(os.path.join(home, ".mujoco", "mujoco210", "bin", "compile"))
            candidates.append(os.path.join(home, ".mujoco", "bin", "compile"))
            candidates.append("/usr/local/bin/compile")

        elif system == "Darwin":  # macOS
            candidates.append("/Applications/MuJoCo.app/Contents/MacOS/compile")

        for c in candidates:
            if os.path.exists(c):
                return c

        raise FileNotFoundError(f"MuJoCo 'compile' binary not found. Searched in PATH and: {', '.join(candidates)}")

    def run(self) -> None:
        """Main execution flow."""
        if not self.urdf_path.exists():
            print(f"Error: URDF not found at {self.urdf_path}")
            return

        # Check for existing output
        if self.xml_path.exists():
            try:
                choice = input(f"Output file {self.xml_path} already exists. Overwrite? [y/N]: ").strip().lower()
                if choice != "y":
                    print("Operation aborted by user.")
                    return
                self.clean_artifacts()
            except EOFError:
                print(f"Output file {self.xml_path} already exists. Non-interactive mode detected. Aborting.")
                return

        # 1. Decompose Collisions
        processing_urdf_path = str(self.urdf_path)
        if self.decompose:
            if coacd is None:
                print("CoACD not installed. Skipping decomposition.")
            else:
                processing_urdf_path = self.process_urdf_collisions(str(self.urdf_path), str(self.mesh_dir))

        # 2. Calculate Height
        print("Calculating auto-height...")
        min_z = self.calculate_min_z(processing_urdf_path)
        height_offset = -min_z + self.safety_margin
        print(f"Lowest point: {min_z:.4f}m. Applied offset: {height_offset:.4f}m (margin: {self.safety_margin}m)")

        # 3. Preprocess URDF
        tmp_urdf = self.preprocess_urdf(processing_urdf_path, height_offset)

        # 4. Compile to XML
        print(f"Compiling to {self.xml_path}...")

        # Find MuJoCo compiler
        try:
            mujoco_bin = self._find_mujoco_binary()
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return

        cmd = [mujoco_bin, tmp_urdf, str(self.xml_path)]
        try:
            subprocess.check_call(cmd)
        except subprocess.CalledProcessError as e:
            print(f"Compilation failed: {e}")
            return
        finally:
            if os.path.exists(tmp_urdf):
                os.remove(tmp_urdf)
            if self.decompose and processing_urdf_path != str(self.urdf_path) and os.path.exists(processing_urdf_path):
                os.remove(processing_urdf_path)

        # 5. Post-process XML
        self.postprocess_xml(str(self.xml_path), height_offset)

        print("\nCompilation and post-processing complete.")


def main():
    parser = argparse.ArgumentParser(description="Compile URDF to MuJoCo XML using URDF2MJCFConverter.")
    parser.add_argument("--target", type=str, default="ace_leader", help="Target robot name.")
    parser.add_argument("--mujoco-bin", type=str, default=None, help="Path to compile.exe.")
    parser.add_argument("--floating", action="store_true", help="Add floating joint.")
    parser.add_argument("--decompose", action="store_true", help="Perform convex decomposition.")
    parser.add_argument("--safety-margin", type=float, default=0.05, help="Extra height margin.")
    parser.add_argument("--q0", type=str, default="", help="Initial joint positions (key=val,key=val).")

    args = parser.parse_args()

    converter = URDF2MJCFConverter(
        target=args.target,
        floating=args.floating,
        decompose=args.decompose,
        safety_margin=args.safety_margin,
        q0=args.q0,
        mujoco_bin=args.mujoco_bin,
    )
    converter.run()


if __name__ == "__main__":
    main()
