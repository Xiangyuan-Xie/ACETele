#include <ace_px4_xrce/arm_joint_state.h>

#include <math.h>
#include <string.h>

static uint32_t read_u32_le(
        const uint8_t* data)
{
    return (uint32_t)data[0] |
           ((uint32_t)data[1] << 8U) |
           ((uint32_t)data[2] << 16U) |
           ((uint32_t)data[3] << 24U);
}

static float read_float_le(
        const uint8_t* data)
{
    const uint32_t bits = read_u32_le(data);
    float result = 0.0F;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

ace_px4_sample_result ace_px4_validate_arm_joint_state(
        const uint8_t* payload,
        size_t size,
        bool has_previous_sequence,
        uint32_t previous_sequence,
        ace_px4_sample_metadata* metadata)
{
    if (payload == NULL || metadata == NULL || size != ACE_PX4_ARM_JOINT_STATE_SIZE)
    {
        return ACE_PX4_SAMPLE_WRONG_SIZE;
    }

    const uint32_t sequence = read_u32_le(payload + 16U);
    const uint8_t joint_count = payload[20U];
    if (payload[21U] > 1U)
    {
        return ACE_PX4_SAMPLE_NONFINITE;
    }
    const bool velocity_valid = payload[21U] != 0U;
    if (joint_count < ACE_PX4_ARM_MIN_JOINTS || joint_count > ACE_PX4_ARM_MAX_JOINTS)
    {
        return ACE_PX4_SAMPLE_INVALID_COUNT;
    }
    if (has_previous_sequence && (int32_t)(sequence - previous_sequence) <= 0)
    {
        return ACE_PX4_SAMPLE_OUT_OF_ORDER;
    }

    for (size_t index = 0U; index < ACE_PX4_ARM_MAX_JOINTS; ++index)
    {
        const float position = read_float_le(payload + 24U + index * sizeof(float));
        const float velocity = read_float_le(payload + 80U + index * sizeof(float));
        if (index < joint_count)
        {
            if (!isfinite(position) || !isfinite(velocity))
            {
                return ACE_PX4_SAMPLE_NONFINITE;
            }
        }
        else if (position != 0.0F || velocity != 0.0F)
        {
            return ACE_PX4_SAMPLE_NONZERO_PADDING;
        }
    }
    if (payload[22U] != 0U || payload[23U] != 0U)
    {
        return ACE_PX4_SAMPLE_NONZERO_PADDING;
    }

    metadata->sequence = sequence;
    metadata->joint_count = joint_count;
    metadata->velocity_valid = velocity_valid;
    return ACE_PX4_SAMPLE_VALID;
}

const char* ace_px4_sample_result_string(
        ace_px4_sample_result result)
{
    switch (result)
    {
        case ACE_PX4_SAMPLE_VALID:
            return "valid";
        case ACE_PX4_SAMPLE_WRONG_SIZE:
            return "wrong payload size";
        case ACE_PX4_SAMPLE_INVALID_COUNT:
            return "joint_count is outside [4, 14]";
        case ACE_PX4_SAMPLE_NONFINITE:
            return "active arm state contains a non-finite value";
        case ACE_PX4_SAMPLE_NONZERO_PADDING:
            return "unused arm state slots are not zero";
        case ACE_PX4_SAMPLE_OUT_OF_ORDER:
            return "sequence is not newer than the previous sample";
        default:
            return "unknown validation result";
    }
}
