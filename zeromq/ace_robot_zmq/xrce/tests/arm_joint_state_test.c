#include <ace_px4_xrce/arm_joint_state.h>

#include <math.h>
#include <stdint.h>
#include <string.h>

#define CHECK(condition) do { if (!(condition)) { return __LINE__; } } while (false)

static void write_u32_le(
        uint8_t* destination,
        uint32_t value)
{
    destination[0] = (uint8_t)value;
    destination[1] = (uint8_t)(value >> 8U);
    destination[2] = (uint8_t)(value >> 16U);
    destination[3] = (uint8_t)(value >> 24U);
}

static void write_float_le(
        uint8_t* destination,
        float value)
{
    uint32_t bits = 0U;
    memcpy(&bits, &value, sizeof(bits));
    write_u32_le(destination, bits);
}

static void valid_payload(
        uint8_t* payload,
        uint32_t sequence,
        uint8_t count)
{
    memset(payload, 0, ACE_PX4_ARM_JOINT_STATE_SIZE);
    write_u32_le(payload + 16U, sequence);
    payload[20U] = count;
    payload[21U] = 1U;
    for (uint8_t index = 0U; index < count; ++index)
    {
        write_float_le(payload + 24U + index * 4U, (float)index * 0.1F);
        write_float_le(payload + 80U + index * 4U, (float)index * 0.2F);
    }
}

int main(void)
{
    uint8_t payload[ACE_PX4_ARM_JOINT_STATE_SIZE];
    ace_px4_sample_metadata metadata;
    valid_payload(payload, 8U, 4U);
    CHECK(ace_px4_validate_arm_joint_state(
              payload, sizeof(payload), false, 0U, &metadata) == ACE_PX4_SAMPLE_VALID);
    CHECK(metadata.sequence == 8U);
    CHECK(metadata.joint_count == 4U);
    CHECK(metadata.velocity_valid);

    valid_payload(payload, 9U, 7U);
    CHECK(ace_px4_validate_arm_joint_state(
              payload, sizeof(payload), true, 8U, &metadata) == ACE_PX4_SAMPLE_VALID);
    valid_payload(payload, 10U, 14U);
    CHECK(ace_px4_validate_arm_joint_state(
              payload, sizeof(payload), true, 9U, &metadata) == ACE_PX4_SAMPLE_VALID);

    CHECK(ace_px4_validate_arm_joint_state(
              payload, sizeof(payload), true, 10U, &metadata) == ACE_PX4_SAMPLE_OUT_OF_ORDER);
    payload[20U] = 3U;
    CHECK(ace_px4_validate_arm_joint_state(
              payload, sizeof(payload), false, 0U, &metadata) == ACE_PX4_SAMPLE_INVALID_COUNT);
    payload[20U] = 15U;
    CHECK(ace_px4_validate_arm_joint_state(
              payload, sizeof(payload), false, 0U, &metadata) == ACE_PX4_SAMPLE_INVALID_COUNT);

    valid_payload(payload, 11U, 4U);
    write_float_le(payload + 24U, NAN);
    CHECK(ace_px4_validate_arm_joint_state(
              payload, sizeof(payload), false, 0U, &metadata) == ACE_PX4_SAMPLE_NONFINITE);

    valid_payload(payload, 12U, 4U);
    write_float_le(payload + 24U + 5U * 4U, 1.0F);
    CHECK(ace_px4_validate_arm_joint_state(
              payload, sizeof(payload), false, 0U, &metadata) ==
          ACE_PX4_SAMPLE_NONZERO_PADDING);

    valid_payload(payload, 13U, 4U);
    payload[22U] = 1U;
    CHECK(ace_px4_validate_arm_joint_state(
              payload, sizeof(payload), false, 0U, &metadata) ==
          ACE_PX4_SAMPLE_NONZERO_PADDING);

    valid_payload(payload, 14U, 4U);
    payload[21U] = 0U;
    write_float_le(payload + 80U, INFINITY);
    CHECK(ace_px4_validate_arm_joint_state(
              payload, sizeof(payload), false, 0U, &metadata) == ACE_PX4_SAMPLE_NONFINITE);
    return 0;
}
