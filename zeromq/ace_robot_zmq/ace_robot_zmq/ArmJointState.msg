# Arm-only joint state for internal RL control.
# Active joints are densely packed in configured arm assembly order. End
# effectors are excluded, and unused array entries are zero-filled.
#
# TOPICS arm_joint_state

uint8 MAX_JOINTS = 14

uint64 timestamp
uint64 timestamp_sample
uint32 sequence
uint8 joint_count
# True when all active entries in arm_velocity contain measured or estimated velocity.
bool arm_velocity_valid
float32[14] arm_position
float32[14] arm_velocity
