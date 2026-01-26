import math
from dataclasses import dataclass


@dataclass
class ServoData:
    """Data class containing essential servo motor specifications."""

    no_load_time_sec_per_60deg: float
    no_load_current_a: float
    stall_torque_kgcm: float
    stall_current_a: float
    rated_load_kgcm: float
    rated_current_a: float
    kt_kgcm_per_a: float


def kgcm_to_Nm(kgcm: float) -> float:
    """Convert torque from kg·cm to N·m."""
    return kgcm * 0.0980665


def sec_per_60deg_to_rad_per_s(sec_per_60deg: float) -> float:
    """Convert rotational speed from seconds per 60 degrees to rad/s."""
    return (math.pi / 3.0) / sec_per_60deg


def estimate(servo: ServoData):
    """
    Estimate servo motor parameters for URDF dynamics with Kt verification.

    Args:
        servo: ServoData object containing servo specifications

    Returns:
        Dictionary with calculated parameters and Kt verification
    """
    # Convert units
    tau_stall_Nm = kgcm_to_Nm(servo.stall_torque_kgcm)
    omega_no_load_rad_s = sec_per_60deg_to_rad_per_s(servo.no_load_time_sec_per_60deg)

    # ===================================================================
    # Kt Verification Section
    # ===================================================================
    # 1. Calculate Kt from stall characteristics
    kt_from_stall_kgcm_per_a = servo.stall_torque_kgcm / servo.stall_current_a
    kt_from_stall_error = abs(kt_from_stall_kgcm_per_a - servo.kt_kgcm_per_a) / servo.kt_kgcm_per_a * 100

    # 2. Calculate Kt from rated load characteristics
    kt_from_rated_kgcm_per_a = servo.rated_load_kgcm / servo.rated_current_a
    kt_from_rated_error = abs(kt_from_rated_kgcm_per_a - servo.kt_kgcm_per_a) / servo.kt_kgcm_per_a * 100

    # 3. Calculate torque constant range
    kt_range_min = min(servo.kt_kgcm_per_a, kt_from_stall_kgcm_per_a, kt_from_rated_kgcm_per_a)
    kt_range_max = max(servo.kt_kgcm_per_a, kt_from_stall_kgcm_per_a, kt_from_rated_kgcm_per_a)
    kt_range_ratio = kt_range_max / kt_range_min if kt_range_min > 0 else float("inf")

    # 4. Check Kt consistency
    kt_consistency_warning = ""
    if kt_from_stall_error > 10:
        kt_consistency_warning = "WARNING: Significant discrepancy between provided Kt and stall-derived Kt"
    elif kt_range_ratio > 1.5:
        kt_consistency_warning = "WARNING: High variation in Kt values calculated from different data points"

    # 5. Recommended Kt value
    kt_recommended_kgcm_per_a = (servo.kt_kgcm_per_a + kt_from_stall_kgcm_per_a + kt_from_rated_kgcm_per_a) / 3
    kt_recommended_Nm_per_A = kgcm_to_Nm(kt_recommended_kgcm_per_a)

    # ===================================================================
    # Servo Dynamics Parameter Estimation
    # ===================================================================
    # 1. Calculate Coulomb Friction
    friction_measured = servo.no_load_current_a * kt_recommended_Nm_per_A

    # 2. Calculate Equivalent Viscous Damping
    damping_total = tau_stall_Nm / omega_no_load_rad_s

    # 3. Rated torque reference
    tau_rated_Nm = kgcm_to_Nm(servo.rated_load_kgcm)

    return {
        "tau_stall_Nm": tau_stall_Nm,
        "omega_rad_s": omega_no_load_rad_s,
        "friction_Nm": friction_measured,
        "damping_Nm_s_rad": damping_total,
        "friction_vs_rated_ratio": friction_measured / tau_rated_Nm if tau_rated_Nm > 0 else float("inf"),
        # Kt verification results
        "kt_provided_kgcm_per_a": servo.kt_kgcm_per_a,
        "kt_from_stall_kgcm_per_a": kt_from_stall_kgcm_per_a,
        "kt_from_rated_kgcm_per_a": kt_from_rated_kgcm_per_a,
        "kt_stall_error_percent": kt_from_stall_error,
        "kt_rated_error_percent": kt_from_rated_error,
        "kt_recommended_kgcm_per_a": kt_recommended_kgcm_per_a,
        "kt_consistency_warning": kt_consistency_warning,
    }


# Model 3950
# servo = ServoData(
#     no_load_time_sec_per_60deg=0.133,  # sec/60deg
#     no_load_current_a=0.33,
#     stall_torque_kgcm=50.0,
#     stall_current_a=2.4,
#     rated_load_kgcm=12.5,
#     rated_current_a=0.6,
#     kt_kgcm_per_a=20.8
# )

# Model 3930
# servo = ServoData(
#     no_load_time_sec_per_60deg=0.222,  # sec/60deg
#     no_load_current_a=0.15,
#     stall_torque_kgcm=35.0,
#     stall_current_a=2.8,
#     rated_load_kgcm=8.7,
#     rated_current_a=0.8,
#     kt_kgcm_per_a=12.5,
# )

# Model 3915
servo = ServoData(
    no_load_time_sec_per_60deg=0.1,
    no_load_current_a=0.26,
    stall_torque_kgcm=14.2,
    stall_current_a=1.5,
    rated_load_kgcm=4.5,
    rated_current_a=0.5,
    kt_kgcm_per_a=9.3,
)

results = estimate(servo)

# Print results
print("=== Servo Physical Parameter Estimation ===")
print("\n--- Torque Constant (Kt) Verification ---")
print(f"1. Provided Kt: {results['kt_provided_kgcm_per_a']:.2f} kg·cm/A")
print(
    f"2. Stall-derived Kt: {results['kt_from_stall_kgcm_per_a']:.2f} kg·cm/A "
    f"(Error: {results['kt_stall_error_percent']:.1f}%)"
)
print(
    f"3. Rated-derived Kt: {results['kt_from_rated_kgcm_per_a']:.2f} kg·cm/A "
    f"(Error: {results['kt_rated_error_percent']:.1f}%)"
)
print(f"4. Recommended Kt: {results['kt_recommended_kgcm_per_a']:.2f} kg·cm/A")

if results["kt_consistency_warning"]:
    print(f"\n⚠️  {results['kt_consistency_warning']}")

print("\n--- Dynamics Parameters ---")
print(f"1. Viscous Damping: {results['damping_Nm_s_rad']:.6f} N·m·s/rad")
print(f"2. Coulomb Friction: {results['friction_Nm']:.6f} N·m")
print(f"3. Friction to Rated Load Ratio: {results['friction_vs_rated_ratio'] * 100:.1f}%")
