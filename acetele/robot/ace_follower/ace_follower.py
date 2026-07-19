from acetele.robot.joint_robot import JointRobot


class AceFollowerRobot(JointRobot):
    """Single-arm ACE follower with an optional configured end effector."""

    ROBOT_TYPE = "ace_follower"


class AceDualFollowerRobot(JointRobot):
    """Dual-arm ACE follower composed from two named arm assemblies."""

    ROBOT_TYPE = "ace_follower_dual"


__all__ = ["AceDualFollowerRobot", "AceFollowerRobot"]
