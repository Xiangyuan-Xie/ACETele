from pymavlink import mavutil


class PX4Interface:
    def __init__(self, host="127.0.0.1", port=4560):
        self._host = host
        self._port = port
        self._mavlink_connection = None
        self._is_connected = False
        self._is_armed = False

        self._initialize_connection()

    @property
    def is_connected(self):
        return self._is_connected

    @property
    def is_armed(self):
        return self._is_armed

    def _initialize_connection(self):
        """Start listening for PX4 connection in server mode."""
        # tcpin:0.0.0.0:4560 listens on all interfaces (server mode).
        # PX4 (client) connects to this process.
        print(f"[PX4 SITL] Listening on tcpin:0.0.0.0:{self._port}...")
        self._mavlink_connection = mavutil.mavlink_connection(
            f"tcpin:{self._host}:{self._port}", source_system=254, source_component=97
        )

    def update_connection_state(self):
        """Update connection state from PX4 heartbeat."""
        if self._is_connected:
            return True

        if self._mavlink_connection:
            msg = self._mavlink_connection.recv_match(type="HEARTBEAT", blocking=False)
            if msg:
                self._is_connected = True
                print("-" * 50)
                print("[PX4 SITL] Connected!")
                print("[PX4 SITL] QGC should auto-connect via UDP 14550.")
                print("-" * 50)
                return True

        return False

    def send_hil_sensor(
        self,
        timestamp_us,
        accel_frd,
        gyro_frd,
        mag_frd,
        altitude_m,
        temperature_celsius=25.0,
        fields_updated=None,
    ):
        """
        Send HIL_SENSOR message to PX4.

        :param timestamp_us: Timestamp in microseconds
        :param accel_frd: [ax, ay, az] in m/s^2 (body frame, FRD)
        :param gyro_frd: [gx, gy, gz] in rad/s (body frame, FRD)
        :param mag_frd: [mx, my, mz] in Gauss (body frame, FRD)
        :param altitude_m: Altitude in meters
        :param temperature_celsius: Temperature in Celsius
        :param fields_updated: Optional bitmask for updated fields (default: calculate based on inputs)
        """
        if not self._is_connected:
            return

        # Fields updated flags:
        # XACC(1) | YACC(2) | ZACC(4) = 7
        # XGYRO(8) | YGYRO(16) | ZGYRO(32) = 56
        # XMAG(64) | YMAG(128) | ZMAG(256) = 448
        # ABS_PRESSURE(512) | DIFF_PRESSURE(1024) | PRESSURE_ALT(2048) | TEMPERATURE(4096).

        if fields_updated is None:
            # Default: everything except DIFF_PRESSURE (1024), which is often not present.
            # Sum = 8191 - 1024 = 7167 (0x1BFF).
            fields_updated = 0x1BFF

        # PX4 expects specific units:
        # Accel: m/s^2
        # Gyro: rad/s
        # Mag: Gauss
        # Abs pressure: hPa (millibar)
        # Pressure altitude: meters
        # Temperature: Celsius.

        # Calculate approximate pressure from altitude using a standard atmosphere model:
        # P = 1013.25 * (1 - 2.25577e-5 * h)^5.25588.
        abs_pressure = 1013.25 * (1 - 2.25577e-5 * altitude_m) ** 5.25588

        self._mavlink_connection.mav.hil_sensor_send(
            timestamp_us,
            accel_frd[0],
            accel_frd[1],
            accel_frd[2],
            gyro_frd[0],
            gyro_frd[1],
            gyro_frd[2],
            mag_frd[0],
            mag_frd[1],
            mag_frd[2],
            abs_pressure,
            0,  # diff_pressure (unused but set to 0)
            altitude_m,
            temperature_celsius,
            fields_updated,
            0,  # id
        )

    def send_hil_gps(
        self,
        timestamp_us,
        latitude_e7,
        longitude_e7,
        altitude_mm,
        ground_speed_cm_s,
        velocity_north_cm_s,
        velocity_east_cm_s,
        velocity_down_cm_s,
        course_over_ground_cdeg,
        satellites_visible=10,
    ):
        """
        Send HIL_GPS message to PX4.

        :param timestamp_us: Timestamp in microseconds
        :param latitude_e7: Latitude in degrees * 1E7
        :param longitude_e7: Longitude in degrees * 1E7
        :param altitude_mm: Altitude in millimeters (AMSL)
        :param ground_speed_cm_s: Speed in cm/s
        :param velocity_north_cm_s: Velocity North in cm/s
        :param velocity_east_cm_s: Velocity East in cm/s
        :param velocity_down_cm_s: Velocity Down in cm/s
        :param course_over_ground_cdeg: Course over ground in centidegrees
        """
        if not self._is_connected:
            return

        fix_type = 3  # 3D fix.
        eph = 100
        epv = 100

        self._mavlink_connection.mav.hil_gps_send(
            timestamp_us,
            fix_type,
            int(latitude_e7),
            int(longitude_e7),
            int(altitude_mm),
            eph,
            epv,
            int(ground_speed_cm_s),
            int(velocity_north_cm_s),
            int(velocity_east_cm_s),
            int(velocity_down_cm_s),
            int(course_over_ground_cdeg),
            satellites_visible,
        )

    def read_actuator_controls(self):
        """Read actuator controls from HIL_ACTUATOR_CONTROLS messages."""
        if not self._is_connected:
            return None

        if self._mavlink_connection:
            msg = self._mavlink_connection.recv_match(type="HIL_ACTUATOR_CONTROLS", blocking=False)
            if msg:
                return msg.controls

        return None

    def update_arming_state(self):
        """Update and return PX4 arming state."""
        if self._is_connected and self._mavlink_connection:
            self._is_armed = bool(self._mavlink_connection.motors_armed())

        return self._is_armed
