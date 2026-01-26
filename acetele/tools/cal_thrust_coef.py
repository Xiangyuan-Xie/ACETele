import matplotlib.pyplot as plt
import numpy as np


def calculate_thrust_coefficient(rpms, forces, plot=True):
    """
    Calculate thrust coefficient from thrust-RPM data using F = k * ω^2

    Parameters:
    rpms: List of rotational speeds (unit: RPM or rad/s, maintain consistency)
    forces: Corresponding thrust forces (unit: N)
    plot: Whether to plot the fitted curve and data points

    Returns:
    k: Thrust coefficient
    R2: Coefficient of determination (goodness of fit)
    """

    # Convert inputs to numpy arrays
    rpms = np.array(rpms, dtype=float)
    forces = np.array(forces, dtype=float)

    # Validate data length
    if len(rpms) != len(forces):
        raise ValueError("RPM and thrust data must have the same length")

    if len(rpms) < 2:
        raise ValueError("At least 2 data points are required for fitting")

    # Display input data
    print(f"Number of data points: {len(rpms)}")
    print("RPM | Thrust (N)")
    print("-" * 25)
    for rpm, force in zip(rpms, forces):
        print(f"{rpm:9.1f} | {force:8.4f}")

    # Method: Assume F = k * ω^2, transform to linear regression
    # Let X = ω^2, Y = F
    omega_squared = rpms**2

    # Linear regression: Y = k * X
    # Solve for k using least squares method
    k = np.sum(omega_squared * forces) / np.sum(omega_squared**2)

    # Calculate fitted values
    forces_fitted = k * omega_squared

    # Calculate R² (coefficient of determination)
    ss_res = np.sum((forces - forces_fitted) ** 2)  # Residual sum of squares
    ss_tot = np.sum((forces - np.mean(forces)) ** 2)  # Total sum of squares
    R2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1

    # Calculate residuals
    residuals = forces - forces_fitted
    rmse = np.sqrt(np.mean(residuals**2))

    # Display fitting results
    print("\nFitting Results:")
    print(f"Thrust coefficient k = {k:.6e}")
    print(f"Fitting formula: F = {k:.6e} * ω²")
    print(f"Coefficient of determination R² = {R2:.6f}")
    print(f"Root Mean Square Error RMSE = {rmse:.6f} N")

    # Plot results
    if plot:
        plt.figure(figsize=(12, 5))

        # Subplot 1: Original data and fitted curve
        plt.subplot(1, 2, 1)

        # Generate smooth curve
        rpm_smooth = np.linspace(min(rpms), max(rpms), 200)
        force_smooth = k * (rpm_smooth**2)

        plt.scatter(rpms, forces, color="red", s=50, label="Experimental Data", zorder=5)
        plt.plot(rpm_smooth, force_smooth, "b-", linewidth=2, label=f"Fitted Curve: F = {k:.2e}ω²")
        plt.xlabel("Rotational Speed (RPM)", fontsize=12)
        plt.ylabel("Thrust Force (N)", fontsize=12)
        plt.title("Thrust-RPM Relationship Fitting", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Subplot 2: Residual plot
        plt.subplot(1, 2, 2)
        plt.scatter(rpms, residuals, color="green", s=50)
        plt.axhline(y=0, color="r", linestyle="--", alpha=0.5)
        plt.xlabel("Rotational Speed (RPM)", fontsize=12)
        plt.ylabel("Residuals (N)", fontsize=12)
        plt.title(f"Residual Plot (R² = {R2:.4f})", fontsize=14)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return k, R2, rmse


# Usage example
if __name__ == "__main__":
    # rpms_rpm = [4042, 4469, 4855, 5301, 5780, 6298, 6800, 7281, 7679, 8096, 8468, 8867, 9257, 9675, 9857]  # RPM
    # thrust_g = [210, 259, 309, 373, 447, 536, 628, 729, 814, 906, 993, 1087, 1191, 1289, 1332]  # g
    # forces_n = [weight * 9.80665 / 1000 for weight in thrust_g]  # N

    rpms_rpm = [3900, 4600, 5100, 5600, 6000]  # RPM
    thrust_g = [830, 1150, 1430, 1690, 1920]  # g
    forces_n = [weight * 9.80665 / 1000 for weight in thrust_g]  # N

    print("=" * 50)
    k1, R2_1, rmse1 = calculate_thrust_coefficient(rpms_rpm, forces_n)
