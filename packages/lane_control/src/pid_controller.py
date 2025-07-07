class PIDController:
    def __init__(self, kp, ki, kd, dt, output_limits=(None, None), integrator_limits=(None, None)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt

        self.output_limits = output_limits
        self.integrator_limits = integrator_limits

        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error):
        # Proportional
        P = self.kp * error

        # Integral with clamping
        self.integral += error * self.dt
        min_i, max_i = self.integrator_limits
        if min_i is not None:
            self.integral = max(min_i, self.integral)
        if max_i is not None:
            self.integral = min(max_i, self.integral)
        I = self.ki * self.integral

        # Derivative
        derivative = (error - self.prev_error) / self.dt
        D = self.kd * derivative
        self.prev_error = error

        # Total output
        output = P + I + D
        min_o, max_o = self.output_limits
        if min_o is not None:
            output = max(min_o, output)
        if max_o is not None:
            output = min(max_o, output)

        return output
