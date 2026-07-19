from acetele.robot.joint_robot import JointRobot


class AceLeaderRobot(JointRobot):
    """Single-arm ACE leader composed from its typed robot configuration."""

    ROBOT_TYPE = "ace_leader"


__all__ = ["AceLeaderRobot"]
