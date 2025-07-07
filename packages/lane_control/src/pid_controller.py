class PIDController:
    def __init__(self, kp_d, ki_d, kp_phi, ki_phi, dt,
                 d_limits=(None, None), phi_limits=(None, None), output_limits=(None, None)):

        self.dt = dt
        self.kp_d = kp_d
        self.ki_d = ki_d
        self.kp_phi = kp_phi
        self.ki_phi = ki_phi

        self.d_integral = 0.0
        self.phi_integral = 0.0

        self.d_limits = d_limits
        self.phi_limits = phi_limits
        self.output_limits = output_limits

        self.last_d_err = 0.0
        self.last_phi_err = 0.0

    def reset(self):
        self.d_integral = 0.0
        self.phi_integral = 0.0

    def update(self, d_err, phi_err):
        # Integral
        self.d_integral += d_err * self.dt
        self.phi_integral += phi_err * self.dt

        # Clamp integral
        self.d_integral = np.clip(self.d_integral, *self.d_limits)
        self.phi_integral = np.clip(self.phi_integral, *self.phi_limits)

        # PID terms
        omega_d = self.kp_d * d_err + self.ki_d * self.d_integral
        omega_phi = self.kp_phi * phi_err + self.ki_phi * self.phi_integral

        omega = omega_d + omega_phi

        # Clamp total output
        omega = np.clip(omega, *self.output_limits)
        return omega

    def update_parameters(self, params):
        self.kp_d = params["~k_d"].value
        self.ki_d = params["~k_Id"].value
        self.kp_phi = params["~k_theta"].value
        self.ki_phi = params["~k_Iphi"].value
