#ifndef ACE_PX4_XRCE_ARM_JOINT_STATE_H
#define ACE_PX4_XRCE_ARM_JOINT_STATE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

enum
{
    ACE_PX4_ARM_MIN_JOINTS = 4,
    ACE_PX4_ARM_MAX_JOINTS = 14,
    ACE_PX4_ARM_JOINT_STATE_SIZE = 136
};

typedef enum ace_px4_sample_result
{
    ACE_PX4_SAMPLE_VALID = 0,
    ACE_PX4_SAMPLE_WRONG_SIZE,
    ACE_PX4_SAMPLE_INVALID_COUNT,
    ACE_PX4_SAMPLE_NONFINITE,
    ACE_PX4_SAMPLE_NONZERO_PADDING,
    ACE_PX4_SAMPLE_OUT_OF_ORDER
} ace_px4_sample_result;

typedef struct ace_px4_sample_metadata
{
    uint32_t sequence;
    uint8_t joint_count;
    bool velocity_valid;
} ace_px4_sample_metadata;

ace_px4_sample_result ace_px4_validate_arm_joint_state(
        const uint8_t* payload,
        size_t size,
        bool has_previous_sequence,
        uint32_t previous_sequence,
        ace_px4_sample_metadata* metadata);

const char* ace_px4_sample_result_string(ace_px4_sample_result result);

#ifdef __cplusplus
}
#endif

#endif
