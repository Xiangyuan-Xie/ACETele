import time

from pymavlink import mavutil


class PX4Interface:
    def __init__(self, host="127.0.0.1", port=4560, verbose=False):
        self.host = host
        self.port = port
        self.verbose = verbose
        self.connection = None
        self.connected = False
        self.last_timestamp = 0
        self.target_system = 1
        self.target_component = 1
        self.qgc_udp = None

        # Start listening for PX4
        self.start_listening()

    def start_listening(self):
        """Start listening for PX4 connection (Server mode)"""
        # tcpin:0.0.0.0:4560 listens on all interfaces (server mode)
        # PX4 (client) will connect to us
        connection_string = f"tcpin:0.0.0.0:{self.port}"
        print(f"[PX4 SITL] Listening on {connection_string}...")
        try:
            self.connection = mavutil.mavlink_connection(connection_string, source_system=254, source_component=97)
        except Exception as e:
            print(f"Failed to start listening: {e}")

    def check_connection(self):
        """Check if PX4 has connected (received heartbeat)"""
        if self.connected:
            return True

        if self.connection is None:
            return False

        # Check for heartbeat (non-blocking)
        msg = self.connection.recv_match(type="HEARTBEAT", blocking=False)
        if msg:
            self.target_system = msg.get_srcSystem()
            self.target_component = msg.get_srcComponent()
            self.connected = True
            if self.qgc_udp is None:
                try:
                    self.qgc_udp = mavutil.mavlink_connection(
                        "udpout:127.0.0.1:14550",
                        source_system=self.target_system,
                        source_component=self.target_component,
                    )
                except Exception:
                    self.qgc_udp = None
            print("-" * 50)
            print(f"[PX4 SITL] Connected! (Sys: {self.target_system}, Comp: {self.target_component})")
            print("[PX4 SITL] QGC should auto-connect via UDP 14550.")
            print("-" * 50)
            return True

        return False

    def set_param(self, name: str, value, confirm: bool = True, retries: int = 20, timeout_s: float = 0.1) -> bool:
        """
        Set a PX4 parameter via MAVLink PARAM_SET and optionally confirm with PARAM_VALUE.
        Returns True if the parameter is confirmed set, else False.
        """
        if not self.connected or self.connection is None:
            return False

        # Determine MAV_PARAM_TYPE; PX4 parameters are commonly REAL32
        try:
            from pymavlink.dialects.v20.common import MAV_PARAM_TYPE

            if isinstance(value, float):
                param_type = MAV_PARAM_TYPE.MAV_PARAM_TYPE_REAL32
                param_value = float(value)
            elif isinstance(value, int):
                # Use REAL32 for broad compatibility
                param_type = MAV_PARAM_TYPE.MAV_PARAM_TYPE_REAL32
                param_value = float(value)
            else:
                # Fallback to REAL32
                param_type = MAV_PARAM_TYPE.MAV_PARAM_TYPE_REAL32
                param_value = float(value)
        except Exception:
            param_type = 9  # MAV_PARAM_TYPE_REAL32
            param_value = float(value)

        name = name[:16]  # MAVLink param_id max length is 16

        for _ in range(max(1, retries)):
            self.connection.mav.param_set_send(
                self.target_system,
                self.target_component,
                name.encode("utf-8"),
                param_value,
                param_type,
            )
            if not confirm:
                return True

            # Wait for matching PARAM_VALUE
            msg = self.connection.recv_match(type="PARAM_VALUE", blocking=True, timeout=timeout_s)
            if msg and msg.get_type() == "PARAM_VALUE":
                if msg.param_id.decode(errors="ignore").strip("\x00") == name:
                    # Allow small tolerance for float rounding
                    if abs(float(msg.param_value) - param_value) < 1e-3:
                        return True
        return False

    def send_hil_sensor(self, time_us, accel, gyro, mag, pressure_alt, temp=25.0, fields_updated=None):
        """
        Send HIL_SENSOR message

        :param time_us: Timestamp in microseconds
        :param accel: [ax, ay, az] in m/s^2 (body frame)
        :param gyro: [gx, gy, gz] in rad/s (body frame)
        :param mag: [mx, my, mz] in Gauss (body frame)
        :param pressure_alt: Altitude in meters
        :param temp: Temperature in Celsius
        :param fields_updated: Optional bitmask for updated fields (default: calculate based on inputs)
        """
        if not self.connected:
            return

        # Fields updated flags
        # XACC(1) | YACC(2) | ZACC(4) = 7
        # XGYRO(8) | YGYRO(16) | ZGYRO(32) = 56
        # XMAG(64) | YMAG(128) | ZMAG(256) = 448
        # ABS_PRESSURE(512) | DIFF_PRESSURE(1024) | PRESSURE_ALT(2048) | TEMPERATURE(4096)

        if fields_updated is None:
            # Default: Everything except DIFF_PRESSURE (1024) which is often not present
            # Sum = 8191 - 1024 = 7167 (0x1BFF)
            fields_updated = 0x1BFF

        # PX4 expects specific units.
        # Accel: m/s^2
        # Gyro: rad/s
        # Mag: Gauss
        # Abs Pressure: hPa (millibar)
        # Pressure Alt: meters
        # Temp: Celsius

        # Calculate approximate pressure from altitude
        # Standard atmosphere: P = P0 * (1 - L*h/T0)^(g*M/(R*L))
        # Simplified: P = 1013.25 * (1 - 2.25577e-5 * h)^5.25588
        abs_pressure = 1013.25 * (1 - 2.25577e-5 * pressure_alt) ** 5.25588

        self.connection.mav.hil_sensor_send(
            time_us,
            accel[0],
            accel[1],
            accel[2],
            gyro[0],
            gyro[1],
            gyro[2],
            mag[0],
            mag[1],
            mag[2],
            abs_pressure,
            0,  # diff_pressure (unused but set to 0)
            pressure_alt,
            temp,
            fields_updated,
            0,  # id
        )

    def send_hil_gps(self, time_us, lat, lon, alt, vel, vn, ve, vd, cog, satellites_visible=10):
        """
        Send HIL_GPS message

        :param time_us: Timestamp in microseconds
        :param lat: Latitude in degrees * 1E7
        :param lon: Longitude in degrees * 1E7
        :param alt: Altitude in millimeters (AMSL)
        :param vel: Speed in cm/s
        :param vn: Velocity North in cm/s
        :param ve: Velocity East in cm/s
        :param vd: Velocity Down in cm/s
        :param cog: Course over ground in centidegrees
        """
        if not self.connected:
            return

        fix_type = 3  # 3D fix
        eph = 100  # GPS HDOP horizontal dilution of position in cm (m*100). If unknown, set to: 65535
        epv = 100  # GPS VDOP vertical dilution of position in cm (m*100). If unknown, set to: 65535

        self.connection.mav.hil_gps_send(
            time_us,
            fix_type,
            int(lat),
            int(lon),
            int(alt),
            eph,
            epv,
            int(vel),
            int(vn),
            int(ve),
            int(vd),
            int(cog),
            satellites_visible,
        )

    def receive_controls(self):
        """
        Check for HIL_ACTUATOR_CONTROLS messages
        """
        if not self.connected:
            return None

        # Drain all pending messages and get the latest controls
        latest_controls = None
        while True:
            msg = self.connection.recv_match(type="HIL_ACTUATOR_CONTROLS", blocking=False)
            if msg is None:
                break
            latest_controls = msg.controls  # array of 16 floats, -1..1 or 0..1 depending on config

        return latest_controls

    def send_rc_channels_to_qgc(self, *channels):
        if self.qgc_udp is None:
            return
        t_ms = int(time.perf_counter() * 1000) & 0xFFFFFFFF
        rc_values = [0xFFFF] * 18
        for i, val in enumerate(channels):
            if i < 18:
                rc_values[i] = int(val)
        self.qgc_udp.mav.rc_channels_send(t_ms, len(channels), *rc_values, 100)
