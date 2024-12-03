# Student name: 
import rospy
import numpy as np
from sensor_msgs.msg import Imu, NavSatFix
import alvinxy as axy
import math
from utils import *

from scipy.spatial.transform import Rotation as R 


class EKF_GPS_Imu():
    def __init__(self):
        rospy.init_node('ekf_gps_imu_node', anonymous=True)
        self.lat = 0
        self.lon = 0
        self.lat0 = 0
        self.lon0 = 0
        self.first_latlon_read = True
        self.dt = None
        self.prev_imu_time = None
        self.prev_gps_time = None

        self.subscription_imu = rospy.Subscriber('/oak/imu', Imu, self.callback_imu)
        self.subscription_gps = rospy.Subscriber('/gps/pvt', NavSatFix, self.callback_gps)
        
        #measurement vector = [gx, gy, gz, ax, ay, az, x, y, z]
        self.measure = np.zeros(9)

        # state  = [x, y, z, vx, vy, vz, q1, q2, q3, q4]
        self.state = np.zeros(10)
        initial_orientation = R.from_quat(quaternion_from_euler(0, 0, 0)) # initial rotation
        self.state[6:] = initial_orientation.as_quat()   
        self.P = np.diag([30, 30, 30, 3, 3, 3, 0.1, 0.1, 0.1, 0.1])
        self.Q = np.diag([0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.05, 0.05, 0.05, 0.05])
        self.R = np.diag([4, 4, 4])

        self.gravity = 9.801

        self.save_ekf_results = open("./localization_ekf_results.txt", "w+")
        self.save_gps_results = open("./localization_gps_results.txt", "w+")
        self.save_imu_results = open("./localization_imu_results.txt", "w+")

        self.counter = 0

    
    def callback_imu(self,msg):
        current_imu_time = msg.header.stamp.to_sec()
        if self.prev_imu_time:
            self.dt = current_imu_time - self.prev_imu_time
        self.prev_imu_time = current_imu_time

        self.measure[0] = np.clip(msg.angular_velocity.x,-5,5) #(-5,5)
        self.measure[1] = np.clip(msg.angular_velocity.y,-5,5) #(-5,5)
        self.measure[2] = np.clip(msg.angular_velocity.z,-5,5) #(-5,5)
        self.measure[3] = np.clip(msg.linear_acceleration.x,-5,5) #..(-6,6)
        self.measure[4] = np.clip(msg.linear_acceleration.y,-5,5) #..(-6,6)
        self.measure[5] = np.clip(msg.linear_acceleration.z,-16,4) #..(-16,-4)

        if self.dt:
            self.ekf_predict()
            self.save_imu_results.write(f"{self.state[0]}, {self.state[1]}\n")
 
    def callback_gps(self, msg):
        self.counter += 1
        print(self.counter)
        if self.first_latlon_read:
            self.lat0, self.lon0 = msg.latitude, msg.longitude
            self.first_latlon_read = False
        else:
            x, y = axy.ll2xy(msg.latitude, msg.longitude, self.lat0, self.lon0)
            self.measure[6:8] = [x, y]
            self.save_gps_results.write(f"{x}, {y}\n")
            if self.dt:
                self.ekf_update()

    def ekf_predict(self):
        accel, gyro = self.measure[3:6], self.measure[:3]
        self.fx(accel, gyro)
        A = self.jacobian_fx()
        self.P = A@self.P@A.T + self.Q

    def ekf_update(self):
        z_pred = self.Hx()  # Predict the measurement
        H = self.jacobian_Hx()  # Jacobian of the measurement model
        err = self.gps_measurement() - z_pred   # Innovation (residual)
        S = H @ self.P @ H.T + self.R   # Innovation covariance
        K = self.P @ H.T @ np.linalg.inv(S) # Kalman gain
        self.state = self.state + K @ err   # Update state estimate
        self.P = (np.eye(len(self.state)) - K @ H) @ self.P # Update covariance estimate
        self.save_ekf_results.write(f"{self.state[0]}, {self.state[1]}\n")

    #state transition function
    def fx(self, accel, gyro):
        self.update_orientation(gyro)
        accel_world = self.convert_accel_to_world(accel)
        # Update velocity with acceleration
        self.state[3:6] += accel_world * self.dt
        # Update position with velocity
        self.state[:3] += self.state[3:6] * self.dt
        # Drift mitigation (simple high-pass filter)
        self.state[3:6] *= 0.98

    def convert_accel_to_world(self, accel):
        rot = R.from_quat(self.state[6:])
        return rot.apply(accel) + np.array([0,0,self.gravity])
    
    def update_orientation(self, gyro):
        delta_orientation = R.from_rotvec(gyro*self.dt)
        curr_orientaion = R.from_quat(self.state[6:])
        new_orientation = curr_orientaion * delta_orientation
        self.state[6:] = new_orientation.as_quat()
        
    
    def normalize_quaternion(self, q):
        return q / np.linalg.norm(q)

    #jacobian of the process model
    def jacobian_fx(self):
        A = np.eye(10)
        A[0,3], A[1,4], A[2,5] = self.dt, self.dt, self.dt
        return A
    
    # measurement function (GPS provides [x, y, z])
    def Hx(self):
        return self.state[:3]
    
    # jacobian of the measurement model
    def jacobian_Hx(self):
        H = np.zeros((3,10))
        H[0,0], H[1,1], H[2,2] = 1, 1, 1
        return H

    def gps_measurement(self):
        return self.measure[6:]
    
    def __del__(self):
        self.save_ekf_results.close()
        self.save_gps_results.close()
        self.save_ekf_results.close()
        

def main():
    ekf_node = EKF_GPS_Imu()
    rospy.spin()


if __name__ == '__main__':
    main()
