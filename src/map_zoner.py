#!/usr/bin/python2.7
import rospy
import math
import csv
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Float64MultiArray,Int64MultiArray
from std_msgs.msg import Int64
from nav_msgs.msg import Odometry
from rospy.numpy_msg import numpy_msg
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from std_msgs.msg import String
from Quadtree import Target, Rect, QuadTree
import tf
import geometry_msgs.msg
from geometry_msgs.msg import Point, PointStamped
from sensor_msgs.msg import LaserScan
import time
import threading
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64

rospy.init_node("mapzoner")

class map_zoner:
    def __init__(self):
        self.percentage=0
        """self.zones is the number of zones you wish  to divide the free are of the environment into
            self.points_x_y_M contains pose information of all free cells."""
        self.zones = rospy.get_param("zones")
        self.dc = rospy.get_param("desired coverage")

        self.pub = rospy.Publisher("/zones", Float64MultiArray, queue_size=1, latch=True)

        """ startup and shutdown topics to communicate messages to start coverage and stop programs after coverage completion """
        self.pub2 = rospy.Publisher("/startup_shutdown", String, queue_size=1)
        self.pub3 = rospy.Publisher("/shutdown", String, queue_size=1) #
        self.pub4 = rospy.Publisher("/coverage_rate", Float64, queue_size=1)
        """/map topic is subscribed to to get map data that is ued for all clustering, quadtree, transformations 
         and general navigation calculations."""
        self.occ_gridmap = rospy.wait_for_message("/map", OccupancyGrid)
        self.PDA = np.array(self.occ_gridmap.data)
        self.rows = np.arange(len(self.PDA))
        self.columns = 5
        self.m_c = np.zeros((len(self.rows), self.columns))
        self.m_z = np.zeros((self.zones, 8))

        """ self.free_cells contains the indices of map.data where "free" cells (value of 0)
        are located """
        self.free_cells = np.argwhere(self.PDA == 0).flatten()


        """ These are other map information that are used in calculations """
        self.map_resolution = self.occ_gridmap.info.resolution
        self.map_width = self.occ_gridmap.info.width
        self.map_height = self.occ_gridmap.info.height
        self.map_origin_x = self.occ_gridmap.info.origin.position.x
        self.map_origin_y = self.occ_gridmap.info.origin.position.y

        """self.x_y_zones is the variable that is assigned the message for the /zones topic. To send send"""
        self.x_y_zones = Float64MultiArray()

        """self.cells is a list that will contain the the grid location(pixel location) of
        the free cells(value of 0)"""
        self.cells_x_y = []

        """self.domain and self.qtree are used to create  quadtree"""
        self.points_x_y_M = []
        self.domain = Rect((self.map_width) / 2, (self.map_height) / 2, self.map_width, self.map_height)
        self.qtree = QuadTree(self.domain, 4)

        self.listener = tf.TransformListener()
        self.scope = PointStamped()


        """ self.scan is a varible that is assigned LaserScan messages """
        self.scan = LaserScan()

        self.finish = " " #topic to communicae if coverage has been reached
        self.NOT = rospy.get_param("NumberofThreads")

    def get_scan(self, msg):
        self.scan = msg

    """distance between center(robot location) of robot and  a free cell """
    def get_distance(self, center, point):
        distance = math.sqrt((point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2)
        return distance

    """converts angles from radians to degrees"""
    def rad_to_deg(self, scan):
        scan_angles = np.arange(scan.angle_min, scan.angle_max, scan.angle_increment).tolist()
        for i in range(len(scan_angles)):
            scan_angles[i] = (int(math.degrees(scan_angles[i]))+360) % 360
        return scan_angles

    """ transfrom points from map to scan frame for coverage calculation"""
    def transform(self, coord,mat44):
        xyz = tuple(np.dot(mat44, np.array([coord[0],coord[1], 0, 1.0])))[:3]
        r = geometry_msgs.msg.PointStamped()
        r.point = geometry_msgs.msg.Point(*xyz)
        return [r.point.x,r.point.y]

    """These sets of functions help convert map.data information to either pose or 
    grid location(pixel location"""

    def cell_location_to_ind(self, cell_location):
        return int((cell_location[1] * int(self.map_width)) + cell_location[0])

    def cell_location_to_x_y_M(self, cell_location):
        x = (cell_location[0] * self.map_resolution) + self.map_origin_x
        y = (cell_location[1] * self.map_resolution) + self.map_origin_y
        return [x, y]

    def ind_to_cell_location(self, index):
        cell_x = index % self.map_width
        cell_y = math.ceil((index - cell_x) / self.map_width)
        return [cell_x, cell_y]

    def x_y_M_to_cell_location(self, x_y_M):
        cell_x = math.ceil((x_y_M[0] - (self.map_origin_x)) / self.map_resolution)
        cell_y = math.ceil((x_y_M[1] - (self.map_origin_y)) / self.map_resolution)
        return [cell_x, cell_y]

    def get_center(self, zone):
        x = []
        y = []
        for cell in zone:
            x.append(cell[0])
            y.append(cell[1])
        x_average = sum(x) / float(len(zone))
        y_average = sum(y) / float(len(zone))
        return [x_average, y_average]


    """This method creates the quadtree, zones and the matrix (m_z) that stores zone inforamtion"""
    def mapmaker(self):
        rospy.loginfo("Preparing Zones .....")
        for cell in self.free_cells:
            cell_x_y = self.ind_to_cell_location(cell)
            self.cells_x_y.append( cell_x_y)
            self.points_x_y_M.append(self.cell_location_to_x_y_M(cell_x_y))

        """This creates the Quadtree"""
        self.points = [Target(*cell) for cell in self.cells_x_y]
        for point in self.points:
            self.qtree.insert(point)

        fit_array = np.array(self.points_x_y_M)

        """KMEANS method of clustering"""
        zones = KMeans(n_clusters=self.zones, n_init=5, max_iter=10, random_state=None).fit(fit_array)
        self.zone_info = np.zeros((len(zones.labels_), 2))
        self.zone_info[:, 0] = self.free_cells
        self.zone_info[:, 1] = zones.labels_
        src = 0
        while src < self.zones:
            carrier = np.where(zones.labels_ == src)
            self.m_z[src, 0] = zones.cluster_centers_[src][0]
            self.m_z[src, 1] = zones.cluster_centers_[src][1]
            self.m_z[src, 2] = len(carrier[0])
            self.m_z[src, 3] = 0
            self.m_z[src, 4] = 10e-20
            self.m_z[src, 5] = 1e20
            src += 1
        rospy.loginfo("Zones are prepared.")


    """Creates matrix (m_c) that updates cell coverage to memory"""
    def tablemaker(self):
        self.m_c[:, 0] = self.rows
        self.m_c[np.where(self.PDA == 100), 1] = -1
        self.m_c[np.where(self.PDA == -1), 1] = -1

        ind = np.isin(self.m_c[:, 0], self.zone_info[:, 0])
        self.m_c[ind, 1] = self.zone_info[:, 1]
        self.m_c[:, 2] = int(0)
        self.m_c[:, 3] = int(-1)

    """Used by the Coverage_calculator method to calculate coverage"""
    def worker(self, TF_MS_t, query, S_F_origin, scan_angles, scan_ranges_t):
        try:
            for found_point in query:
                index = self.cell_location_to_ind(found_point)
                if self.m_c[index, 2]==0:
                    X_Y_M = self.cell_location_to_x_y_M(found_point)
                    X_Y_S = self.transform(X_Y_M , TF_MS_t)
                    alpha = (int(math.degrees(math.atan2(X_Y_S[1], X_Y_S[0]))) + 360) % 360

                    try:
                        if scan_angles.index(alpha) >= 0: #determine if scan angle matches to cell orientation
                            dist = self.get_distance(X_Y_S, S_F_origin)
                            if dist <= scan_ranges_t[scan_angles.index(alpha)]: #determine if cell distance matches scan range
                                self.m_c[index, 2] = 1
                                #found_point = self.cell_location_to_x_y_M(found_point)
                                #pnt_list = []
                                #pnt_list.append(found_point[0])
                                #pnt_list.append(found_point[1])
                                #where_to = '/home/philip/Desktop/coverage.csv'
                                #with open(where_to, 'a') as csvfile:
                                #    csvwriter = csv.writer(csvfile)
                                #    csvwriter.writerow(pnt_list)
                                src = int(self.m_c[index, 1])
                                self.m_z[src, 3] = self.m_z[src, 3] + 1
                                self.m_z[src, 4] = (float(self.m_z[src, 3]) / self.m_z[src, 2]) * 100

                    except ValueError:
                        continue

        except IndexError:
            pass

    def Coverage_Calculator(self,TF_MS_t, x0, y0, z0, grid_range, scan_angles,scan_ranges_t):
        Cell_X_Y = self.x_y_M_to_cell_location([x0, y0])
        center, radius = [Cell_X_Y[0], Cell_X_Y[1]], grid_range
        query = []
        center = Target(*center)
        query = np.array(self.qtree.query_radius(center, radius, query))
        S_F_origin = [0,0] #sensor frame origin #self.transform([x0,y0],rot_mat)

        if self.Multi_threading:
            scope = []
            FOV = int(len(query) / float(self.NOT))
            for i in range(0,self.NOT):
                if i==self.NOT-1:
                   scope.append(threading.Thread(target=self.worker, args=(TF_MS_t,query[FOV*i:], S_F_origin, scan_angles, scan_ranges_t)))
                   scope[-1].start()
                   scope[-1].join()
                scope.append(threading.Thread(target=self.worker, args=(TF_MS_t,query[FOV*i:FOV*(i+1)], S_F_origin, scan_angles, scan_ranges_t)))
                scope[-1].start()
                scope[-1].join()
        else:
            self.worker(TF_MS_t, query, S_F_origin, scan_angles, scan_ranges_t)

        set_time = rospy.Time(0)
        self.listener.waitForTransform('map', 'base_footprint', set_time, rospy.Duration(600))
        (trans, rot) = self.listener.lookupTransform('map', 'base_footprint', set_time)
        x = trans[0]
        y = trans[1]
        src = 0
        count = 1e20
        mini = min(self.m_z[:, 4])
        mlist = np.where( (self.m_z[:, 4]<=mini) )

        for i in mlist[0]:
            self.m_z[i, 5] = self.get_distance([self.m_z[i, 0], self.m_z[i, 1]], [x, y])
            if self.m_z[i, 5] < count:
                count = self.m_z[i, 5]
                src = i

        self.x_y_zones.data = [self.m_z[src, 0], self.m_z[src, 1]]
        self.pub.publish(self.x_y_zones)
        current_cov_rate = (np.sum(self.m_z[:, 3]) / float(len(self.free_cells))) * 100
        self.pub4.publish(current_cov_rate)

        #if current_cov_rate - self.percentage >= 1:
        #    time_now = time.time()
        #    Time_stamp = time_now - self.begin
        #    Time_stamp = (Time_stamp / 60)
        #    self.begin = time_now
        #    record("increment", Time_stamp)
        #    self.percentage = current_cov_rate

        if round(current_cov_rate) >= self.dc:
            rospy.loginfo("Desired coverage is reached. Chaotic coverage path planner stopping.....")
            self.pub2.publish("stop")
            self.pub3.publish("finished")

    """Used to stop the logistician"""
    def shutdown(self, msg):
        self.finish = msg.data

    """Used to start and stop chaotic motion"""
    def start(self, msg):
        self.pub2.publish(msg)

    """Uses the coverage calculator function to keep coverage updated as the robot moves around the map"""
    def logistician(self):
        scan_angles = self.rad_to_deg(self.scan)
        #grid_range = int(round(rospy.get_param("sensing_range") / self.map_resolution))
        grid_range = int(round(self.scan.range_max / self.map_resolution))
        scan_frame = rospy.get_param("scan_frame")
        self.Multi_threading = rospy.get_param("Multi-threading")

        while self.finish != "finished":
            set_time = rospy.Time(0)
            self.listener.waitForTransform(scan_frame, 'map', set_time, rospy.Duration(600))
            (trans, rot) = self.listener.lookupTransform(scan_frame, 'map', set_time)
            TF_MS_t = np.dot(tf.transformations.translation_matrix(trans), tf.transformations.quaternion_matrix(rot))
            scan = self.scan
            scan_ranges_t = scan.ranges
            self.listener.waitForTransform('map', scan_frame, set_time, rospy.Duration(600))
            (X_Y_S_M_t, rot) = self.listener.lookupTransform('map', scan_frame, set_time) #Get pose of sensor at time t

            x_S_M = X_Y_S_M_t[0]
            y_S_M = X_Y_S_M_t[1]
            z_S_M = 0.0

            self.Coverage_Calculator(TF_MS_t, x_S_M, y_S_M, z_S_M, grid_range, scan_angles, scan_ranges_t)


#def record(x, y):
#    pnt_list = []

#    pnt_list.append(x)
#    pnt_list.append(y)

#    with open('/home/philip/Desktop/record.csv', 'a') as csvfile:
#        csvwriter = csv.writer(csvfile)
#        csvwriter.writerow(pnt_list)

if __name__ == "__main__":
    try:
        map = map_zoner()
        rospy.Subscriber("/scan", LaserScan, map.get_scan)

        """subscribed to topic responsible for stopping Logistician"""
        rospy.Subscriber("/shutdown", String, map.shutdown)
        #begin = time.time()
        map.mapmaker()
        map.tablemaker()
        #ending = time.time()
        #preptime = ending - begin
        #preptime = preptime / 60
        """Uses publisher to start chaotic motion (i.e ATP function)"""
        map.start("start")
        #time.sleep(0.5)
        #begin = time.time()
        #map.begin = begin
        map.logistician()
        #ending = time.time()
        #Time = ending - begin
        #Time = Time / 60
        #record("coveragetime", Time)

    except rospy.ROSInterruptException:
        pass
