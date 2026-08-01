#include <ace_px4_xrce/arm_joint_state.h>

#include <ucdr/microcdr.h>
#include <uxr/client/client.h>

#include <errno.h>
#include <poll.h>
#include <signal.h>
#include <stdbool.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#if UXR_CLIENT_VERSION_MAJOR != 2 || UXR_CLIENT_VERSION_MINOR != 4 || UXR_CLIENT_VERSION_MICRO != 0
#error "ace-px4-xrce-publisher requires Micro XRCE-DDS Client 2.4.0"
#endif

#ifndef ACE_PX4_SCHEMA_SHA256
#error "ACE_PX4_SCHEMA_SHA256 must be provided by the build"
#endif

typedef struct options
{
    const char* agent_host;
    const char* agent_port;
    const char* state_socket;
    const char* ack_socket;
    const char* namespace_name;
    uint32_t client_key;
    uint16_t domain_id;
    uint32_t startup_timeout_ms;
} options;

static volatile sig_atomic_t stop_requested = 0;

static void request_stop(
        int signal_number)
{
    (void)signal_number;
    stop_requested = 1;
}

static uint64_t monotonic_milliseconds(void)
{
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0)
    {
        return 0U;
    }
    return (uint64_t)value.tv_sec * 1000U + (uint64_t)value.tv_nsec / 1000000U;
}

static bool parse_u32(
        const char* value,
        uint32_t minimum,
        uint32_t maximum,
        uint32_t* result)
{
    char* end = NULL;
    errno = 0;
    const unsigned long parsed = strtoul(value, &end, 0);
    if (errno != 0 || end == value || *end != '\0' || parsed < minimum || parsed > maximum)
    {
        return false;
    }
    *result = (uint32_t)parsed;
    return true;
}

static bool parse_options(
        int argc,
        char** argv,
        options* result)
{
    *result = (options){
        .agent_host = NULL,
        .agent_port = NULL,
        .state_socket = NULL,
        .ack_socket = NULL,
        .namespace_name = "",
        .client_key = 0U,
        .domain_id = 0U,
        .startup_timeout_ms = 3000U,
    };
    for (int index = 1; index < argc; index += 2)
    {
        if (index + 1 >= argc)
        {
            return false;
        }
        const char* name = argv[index];
        const char* value = argv[index + 1];
        uint32_t parsed = 0U;
        if (strcmp(name, "--agent-host") == 0)
        {
            result->agent_host = value;
        }
        else if (strcmp(name, "--agent-port") == 0)
        {
            result->agent_port = value;
        }
        else if (strcmp(name, "--state-socket") == 0)
        {
            result->state_socket = value;
        }
        else if (strcmp(name, "--ack-socket") == 0)
        {
            result->ack_socket = value;
        }
        else if (strcmp(name, "--namespace") == 0)
        {
            result->namespace_name = value;
        }
        else if (strcmp(name, "--client-key") == 0 &&
                 parse_u32(value, 1U, UINT32_MAX, &parsed))
        {
            result->client_key = parsed;
        }
        else if (strcmp(name, "--domain-id") == 0 && parse_u32(value, 0U, 232U, &parsed))
        {
            result->domain_id = (uint16_t)parsed;
        }
        else if (strcmp(name, "--startup-timeout-ms") == 0 &&
                 parse_u32(value, 1U, 60000U, &parsed))
        {
            result->startup_timeout_ms = parsed;
        }
        else
        {
            return false;
        }
    }
    return result->agent_host != NULL && result->agent_port != NULL &&
           result->state_socket != NULL && result->ack_socket != NULL &&
           result->client_key != 0U;
}

static bool make_unix_address(
        const char* path,
        struct sockaddr_un* address)
{
    const size_t length = strlen(path);
    if (length == 0U || length >= sizeof(address->sun_path))
    {
        return false;
    }
    memset(address, 0, sizeof(*address));
    address->sun_family = AF_UNIX;
    memcpy(address->sun_path, path, length + 1U);
    return true;
}

static bool send_control(
        int socket_fd,
        const struct sockaddr_un* destination,
        const char* format,
        ...)
{
    char message[256];
    va_list arguments;
    va_start(arguments, format);
    const int length = vsnprintf(message, sizeof(message), format, arguments);
    va_end(arguments);
    if (length < 0 || (size_t)length >= sizeof(message))
    {
        return false;
    }
    return sendto(
               socket_fd,
               message,
               (size_t)length,
               MSG_DONTWAIT,
               (const struct sockaddr*)destination,
               sizeof(*destination)) == length;
}

static bool connect_session_until(
        uxrSession* session,
        uint32_t timeout_ms)
{
    const uint64_t deadline = monotonic_milliseconds() + timeout_ms;
    do
    {
        if (uxr_create_session_retries(session, 1U))
        {
            return true;
        }
    }
    while (!stop_requested && monotonic_milliseconds() < deadline);
    return false;
}

static bool create_entities(
        uxrSession* session,
        uxrStreamId stream,
        uint16_t domain_id,
        const char* namespace_name,
        uxrObjectId* datawriter_id)
{
    char topic_name[192];
    const int topic_length = namespace_name[0] == '\0' ?
            snprintf(topic_name, sizeof(topic_name), "rt/fmu/in/arm_joint_state") :
            snprintf(
                topic_name,
                sizeof(topic_name),
                "rt/%s/fmu/in/arm_joint_state",
                namespace_name);
    if (topic_length < 0 || (size_t)topic_length >= sizeof(topic_name))
    {
        return false;
    }

    const uxrObjectId participant_id = uxr_object_id(0x01, UXR_PARTICIPANT_ID);
    const uxrObjectId topic_id = uxr_object_id(0x01, UXR_TOPIC_ID);
    const uxrObjectId publisher_id = uxr_object_id(0x01, UXR_PUBLISHER_ID);
    *datawriter_id = uxr_object_id(0x01, UXR_DATAWRITER_ID);
    const uxrQoS_t qos = {
        .durability = UXR_DURABILITY_VOLATILE,
        .reliability = UXR_RELIABILITY_BEST_EFFORT,
        .history = UXR_HISTORY_KEEP_LAST,
        .depth = 1U,
    };
    const uint16_t requests[] = {
        uxr_buffer_create_participant_bin(
            session, stream, participant_id, domain_id, "ace_px4_xrce", UXR_REPLACE),
        uxr_buffer_create_topic_bin(
            session,
            stream,
            topic_id,
            participant_id,
            topic_name,
            "px4_msgs::msg::dds_::ArmJointState_",
            UXR_REPLACE),
        uxr_buffer_create_publisher_bin(
            session, stream, publisher_id, participant_id, UXR_REPLACE),
        uxr_buffer_create_datawriter_bin(
            session, stream, *datawriter_id, publisher_id, topic_id, qos, UXR_REPLACE),
    };
    uint8_t statuses[sizeof(requests) / sizeof(requests[0])] = {0U};
    const size_t count = sizeof(requests) / sizeof(requests[0]);
    if (!uxr_run_session_until_all_status(session, 1000, requests, statuses, count))
    {
        return false;
    }
    for (size_t index = 0U; index < count; ++index)
    {
        if (statuses[index] != UXR_STATUS_OK && statuses[index] != UXR_STATUS_OK_MATCHED)
        {
            return false;
        }
    }
    return true;
}

static bool publish_payload(
        uxrSession* session,
        uxrStreamId stream,
        uxrObjectId datawriter_id,
        const uint8_t* payload)
{
    ucdrBuffer buffer;
    if (uxr_prepare_output_stream(
            session,
            stream,
            datawriter_id,
            &buffer,
            ACE_PX4_ARM_JOINT_STATE_SIZE) == 0U)
    {
        return false;
    }
    if (!ucdr_serialize_array_uint8_t(
            &buffer, payload, ACE_PX4_ARM_JOINT_STATE_SIZE))
    {
        return false;
    }
    return uxr_run_session_time(session, 1);
}

int main(
        int argc,
        char** argv)
{
    options arguments;
    if (!parse_options(argc, argv, &arguments))
    {
        fprintf(stderr, "invalid arguments\n");
        return 2;
    }
    signal(SIGINT, request_stop);
    signal(SIGTERM, request_stop);

    struct sockaddr_un state_address;
    struct sockaddr_un ack_address;
    if (!make_unix_address(arguments.state_socket, &state_address) ||
            !make_unix_address(arguments.ack_socket, &ack_address))
    {
        fprintf(stderr, "Unix socket path is invalid\n");
        return 2;
    }

    int result = 1;
    int state_socket = -1;
    int ack_socket = -1;
    bool transport_open = false;
    bool session_open = false;
    uxrUDPTransport transport;
    uxrSession session;
    unlink(arguments.state_socket);
    state_socket = socket(AF_UNIX, SOCK_DGRAM | SOCK_NONBLOCK, 0);
    ack_socket = socket(AF_UNIX, SOCK_DGRAM | SOCK_NONBLOCK, 0);
    if (state_socket < 0 || ack_socket < 0 ||
            bind(state_socket, (const struct sockaddr*)&state_address, sizeof(state_address)) != 0)
    {
        fprintf(stderr, "could not create Unix datagram IPC: %s\n", strerror(errno));
        goto cleanup;
    }
    chmod(arguments.state_socket, S_IRUSR | S_IWUSR);

    if (!uxr_init_udp_transport(
            &transport, UXR_IPv4, arguments.agent_host, arguments.agent_port))
    {
        send_control(ack_socket, &ack_address, "ERROR could not create UDP transport");
        goto cleanup;
    }
    transport_open = true;
    uxr_init_session(&session, &transport.comm, arguments.client_key);
    if (!connect_session_until(&session, arguments.startup_timeout_ms))
    {
        send_control(ack_socket, &ack_address, "ERROR could not create XRCE session");
        goto cleanup;
    }
    session_open = true;

    uint8_t reliable_buffer[UXR_CONFIG_UDP_TRANSPORT_MTU * 4U];
    uint8_t reliable_input_buffer[UXR_CONFIG_UDP_TRANSPORT_MTU * 4U];
    uint8_t best_effort_buffer[UXR_CONFIG_UDP_TRANSPORT_MTU];
    const uxrStreamId reliable_output = uxr_create_output_reliable_stream(
        &session, reliable_buffer, sizeof(reliable_buffer), 4U);
    uxr_create_input_reliable_stream(
        &session, reliable_input_buffer, sizeof(reliable_input_buffer), 4U);
    const uxrStreamId best_effort_output = uxr_create_output_best_effort_stream(
        &session, best_effort_buffer, sizeof(best_effort_buffer));
    uxrObjectId datawriter_id;
    if (!create_entities(
            &session,
            reliable_output,
            arguments.domain_id,
            arguments.namespace_name,
            &datawriter_id))
    {
        send_control(ack_socket, &ack_address, "ERROR could not create XRCE entities");
        goto cleanup;
    }
    if (!send_control(
            ack_socket,
            &ack_address,
            "READY %s %s",
            ACE_PX4_SCHEMA_SHA256,
            UXR_CLIENT_VERSION_STR))
    {
        fprintf(stderr, "could not send readiness acknowledgement\n");
        goto cleanup;
    }

    bool has_previous_sequence = false;
    uint32_t previous_sequence = 0U;
    uint8_t latest_payload[ACE_PX4_ARM_JOINT_STATE_SIZE];
    struct pollfd descriptor = {.fd = state_socket, .events = POLLIN, .revents = 0};
    while (!stop_requested)
    {
        const int poll_result = poll(&descriptor, 1U, 10);
        if (poll_result < 0 && errno != EINTR)
        {
            send_control(ack_socket, &ack_address, "ERROR state IPC poll failed");
            goto cleanup;
        }
        bool received = false;
        ssize_t received_size = 0;
        ssize_t latest_size = 0;
        while ((received_size = recv(
                        state_socket,
                        latest_payload,
                        sizeof(latest_payload),
                        MSG_DONTWAIT | MSG_TRUNC)) >= 0)
        {
            received = true;
            latest_size = received_size;
        }
        if (received_size < 0 && errno != EAGAIN && errno != EWOULDBLOCK)
        {
            send_control(ack_socket, &ack_address, "ERROR state IPC receive failed");
            goto cleanup;
        }
        if (!received)
        {
            if (!uxr_run_session_time(&session, 1))
            {
                send_control(ack_socket, &ack_address, "ERROR XRCE session disconnected");
                goto cleanup;
            }
            continue;
        }

        ace_px4_sample_metadata metadata;
        const ace_px4_sample_result validation = ace_px4_validate_arm_joint_state(
            latest_payload,
            (size_t)latest_size,
            has_previous_sequence,
            previous_sequence,
            &metadata);
        if (validation != ACE_PX4_SAMPLE_VALID)
        {
            send_control(
                ack_socket,
                &ack_address,
                "ERROR invalid ArmJointState: %s",
                ace_px4_sample_result_string(validation));
            goto cleanup;
        }
        if (!publish_payload(&session, best_effort_output, datawriter_id, latest_payload))
        {
            send_control(ack_socket, &ack_address, "ERROR XRCE state write failed");
            goto cleanup;
        }
        has_previous_sequence = true;
        previous_sequence = metadata.sequence;
        if (!send_control(
                ack_socket,
                &ack_address,
                "ACK %u %llu",
                metadata.sequence,
                (unsigned long long)monotonic_milliseconds()))
        {
            fprintf(stderr, "could not send publication acknowledgement\n");
            goto cleanup;
        }
    }
    result = 0;

cleanup:
    if (session_open)
    {
        (void)uxr_delete_session_retries(&session, 1U);
    }
    if (transport_open)
    {
        uxr_close_udp_transport(&transport);
    }
    if (state_socket >= 0)
    {
        close(state_socket);
    }
    if (ack_socket >= 0)
    {
        close(ack_socket);
    }
    unlink(arguments.state_socket);
    return result;
}
