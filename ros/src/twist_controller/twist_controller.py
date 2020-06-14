
GAS_DENSITY = 2.858
ONE_MPH = 0.44704
from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter
import rospy


class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit, accel_limit, wheel_radius, wheel_base, steer_ratio,
                 max_lat_accel, max_steer_angle):

        self.yaw_controller = YawController(wheel_base,steer_ratio,0.1,max_lat_accel,max_steer_angle)

        kp = 0.3
        ki = 0.1
        kd = 0.0
        min_throttle = 0.0
        max_throttle = 0.6

        self.throttle_controller = PID(kp,ki,kd,min_throttle,max_throttle)

        tau = 0.5 # 1/2(pi*tau) = cutoff frequency
        ts = 0.02 #sample time

        self.velocity_lpf = LowPassFilter(tau, ts)
        self.last_velocity = 0

        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius

        self.last_time = rospy.get_time()


    def control(self, dbw_enabled, linear_velocity, angular_velocity, current_velocity ):
        # Return throttle, brake, steer

        if not dbw_enabled:
            self.throttle_controller.reset()
            return 0.0, 0.0, 0.0
        else:
            current_velocity = self.velocity_lpf.filt(current_velocity)
            steering = self.yaw_controller.get_steering(linear_velocity,angular_velocity,current_velocity)
            vel_error = linear_velocity - current_velocity
            self.last_velocity = current_velocity
            current_time = rospy.get_time()
            sample_time = current_time - self.last_time
            throttle = self.throttle_controller.step(vel_error, sample_time)
            brake = 0

            if linear_velocity == 0.0 and current_velocity < 0.1:
                throttle = 0
                #Carla has an automatic transmission, which means the car will roll forward
                #if no brake and no throttle is applied.
                #To prevent Carla from moving requires about 700 Nm of torque.
                brake = 700 #N*m

            elif throttle < 0.1 and vel_error < 0:
                throttle = 0
                decel = max(vel_error, self.decel_limit)
                brake = abs(decel)*self.vehicle_mass*self.wheel_radius
            return throttle, brake, steering
