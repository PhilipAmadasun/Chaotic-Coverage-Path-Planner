#!/usr/bin/python2.7
""" This source code is part of the Chaotic coverage path planning application"""
import rospy
import math
from rospy.numpy_msg import numpy_msg
import numpy as np
import time
from move_base_msgs.msg import MoveBaseActionGoal
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Path
from std_msgs.msg import Float64MultiArray
from Quadtree import Target, Rect, QuadTree
import tf
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String
from std_msgs.msg import Float64
import time
from arnold_RK4 import arnold_RK4


rospy.init_node("controlled_chaotic_trajectory_planner")

class Chaotic_system:
    def __init__(self):
        self.start_stop = "" #Variable used for subscriber to topic that dicates when to start and stop operations
        self.pub = rospy.Publisher("goal", PoseStamped, queue_size=1, latch=True)
        self.count = 0
        self.point = []
        self.occ_gridmap = rospy.wait_for_message("/map", OccupancyGrid)
        self.PDA = np.array(self.occ_gridmap.data) #probability data array inforamtion of occupancygrid map.
        self.free_cells = np.argwhere(self.PDA == 0).flatten() #index values of free cells (from probability data array {PDA}).
        self.map_resolution = self.occ_gridmap.info.resolution
        self.map_width = self.occ_gridmap.info.width
        self.map_height = self.occ_gridmap.info.height
        self.map_origin_x = self.occ_gridmap.info.origin.position.x
        self.map_origin_y = self.occ_gridmap.info.origin.position.y
        self.change = 0
        self.route = np.array([])
        self.x_y_zone = [] #This class attribute be assigned the centroids of zones to spread trajectories to.
        self.goal = PoseStamped()
        self.memory = 0
        self.x = 0
        self.y = 1
        self.z = 0

        self.IC_vec = [] #Vector will contain initial conditions of chaotic system

        self.listener = tf.TransformListener()
        self.scope = PointStamped()

        self.domain = Rect((self.map_width) / 2, (self.map_height) / 2, (self.map_width), (self.map_height))
        self.qtree = QuadTree(self.domain, 4) #self.qt is a quadtree made the location (X,Y) of all the free cells.

        self.stage=1
    #Method will help to start and stop chaotic trajectory generation.
    def call_back(self, msg):
        self.start_stop = msg.data

    def get_distance(self, first, second):
        distance = math.sqrt((second[1] - first[0]) ** 2 + (second[1] - first[1]) ** 2)
        return distance

    """These next 4 methods will provide a means of conversion of coordinates of cells"""
    def x_y_M_to_cell_location(self, x_y_M): #converts point coordinate of cell in the map frame to cell location in occupancy-grid map.
        cell_x = math.ceil((x_y_M[0] - (self.map_origin_x)) / self.map_resolution)
        cell_y = math.ceil((x_y_M[1] - (self.map_origin_y)) / self.map_resolution)
        return [cell_x, cell_y]

    def cell_location_to_ind(self, cell_location): #convert cell location in occupancy-grid map to cells index value in PDA.
        return int((cell_location[1] * int(self.map_width)) + cell_location[0])

    def cell_location_to_x_y_M(self, cell_location):
        x = (cell_location[0] * self.map_resolution) + self.map_origin_x
        y = (cell_location[1] * self.map_resolution) + self.map_origin_y
        return [x, y]

    def ind_to_cell_location(self, index):
        cell_x = index % self.map_width
        cell_y = math.ceil((index - cell_x) / self.map_width)
        return [cell_x, cell_y]

    """This method creates the Quadtree"""
    def mapmaker(self):
        self.cells_x_y = []
        for free_cell in self.free_cells:
            self.cells_x_y.append(self.ind_to_cell_location(free_cell))

        self.points = [Target(*cell) for cell in self.cells_x_y]
        for point in self.points:
            self.qtree.insert(point)

    def choose_marker(self, msg): #method used for subscriber to centroid information.
        self.x_y_zone = [msg.data[0], msg.data[1]]

    def path_watcher(self, msg): #Keeps track of path plan to goal.
        self.route = msg.poses

    """These next 2 methods use cost function for obstacle avoidance"""
    def cost_calculator(self, ecol, erow):
        cell_bucket = []
        subset = []
        l = 6
        for i in range(1, l+1):
            icol = ecol - i
            col = ecol + i
            for j in range(0, l):
                row = erow + j
                irow = erow - j
                cell_bucket.append([col, row])
                cell_bucket.append([col, irow])
                cell_bucket.append([icol, row])
                cell_bucket.append([icol, irow])
        count = 0
        for cell in cell_bucket:
            subset.append(cell)
            try:
                if ecol>=self.map_width or erow>=self.map_height or ecol<=0 or erow<=0:
                    count += 500

                else:
                    count += abs(self.PDA[int((cell[1] * int(self.map_width)) + cell[0])])
            except:
                count += 500
        percentage = count / len(cell_bucket)
        return percentage

    def shift(self, n_TP_DS_R, cell_x, cell_y, prev_tp, radius):
        cost = 10e20
        query = []
        if self.stage == 3:
            query.append([cell_x, cell_y])
        center, radius = [cell_x, cell_y], radius  # 7
        center = Target(*center)
        self.qtree.query_radius(center, radius,  query)

        if len(query) == 0:
            return {"arnpnt": n_TP_DS_R, "cost": 10e20}

        for point in query:
            self.point = self.cell_location_to_x_y_M(point)
            gx = self.cost_calculator(point[0], point[1])
            if self.stage == 1:
                fx = self.get_distance(self.point, prev_tp)
            elif self.stage == 2 or self.stage == 3:
                fx = 0
            Cost_Total = fx + gx
            if Cost_Total==0:
                cost = Cost_Total
                n_TP_DS_R[3] = self.point[0]
                n_TP_DS_R[4] = self.point[1]
                break
            if Cost_Total < cost:
                cost = Cost_Total
                n_TP_DS_R[3] = self.point[0]
                n_TP_DS_R[4] = self.point[1]

        return {"arnpnt": n_TP_DS_R, "cost": cost}


    """This method generates trajectory points by integrating the arnold dynamical system with chaos control techniques"""
    def ArnoldLogistic_coverage(self, A, B, C, v):
        dt = rospy.get_param("dt")
        n_iter = rospy.get_param("n_iter")
        ns = rospy.get_param("ns")
        dist_to_goal = rospy.get_param("dist_to_goal")
        DS_ind = 2 #Index of arnold dynamical system coordinate.
        self.goal.header.stamp = rospy.get_rostime()
        self.goal.header.frame_id = 'map'
        self.goal.pose.position.x = 0
        self.goal.pose.position.y = 0
        self.goal.pose.position.z = 0
        self.goal.pose.orientation.x = 0
        self.goal.pose.orientation.y = 0
        self.goal.pose.orientation.z = 0
        self.goal.pose.orientation.w = 1

        self.memory = 0 #Variable keeps track of the number of iterations of trajectory points generated.
        self.set = "" #Variable helps determine IC for the next iterations of trajectory points.
        self.listener.waitForTransform('map', 'base_footprint', rospy.Time(0), rospy.Duration(0.5))
        (trans, rot) = self.listener.lookupTransform('map', 'base_footprint', rospy.Time(0))
        self.IC_vec = [self.x, self.y, self.z, trans[0], trans[1]] #vecto for initial conditions of arnold dynamical system.
        TP_DS_R = self.IC_vec # This variable is a temporary matrix of Arnold dynamical system and robot coordinates.

        """These next two variables help dictate what whether to move to least covered areas
         if robot is unable to perform coverage around it's immediate location"""
        bad_seed_count = 0
        start = "new"

        while self.start_stop != "stop":
            if self.start_stop == "stop":
                continue

            if self.memory >= n_iter or bad_seed_count == 3:
                self.stage = 3
                start = "new"
                bad_seed_count=0
                self.set = "NEW SET"
                cell_x_y_zone = self.x_y_M_to_cell_location(self.x_y_zone)
                target = self.shift(n_TP_DS_R, cell_x_y_zone[0], cell_x_y_zone[1], [], 19)
                n_TP_DS_R = target["arnpnt"]
                self.goal.pose.position.x = n_TP_DS_R[3]
                self.goal.pose.position.y = n_TP_DS_R[4]
                self.pub.publish(self.goal)
                rospy.sleep(0.5)
                reach = len(self.route)
                hlfway = reach / 2
                ptick = time.time()

                """These segments of code publish goals and dictate when to set new goals,
                 based on information on the path plan from the /move_base/NavfnROS/plan topic."""
                while reach >= dist_to_goal:
                    if reach == 0:
                        break
                    self.pub.publish(self.goal)
                    rospy.sleep(0.5)
                    reach = len(self.route)
                    ptock = time.time()
                    pticktock = ptock - ptick
                    if pticktock > 550 and reach <= hlfway:
                        break
                    if reach <= 80:
                        tick = time.time()
                        while reach > dist_to_goal:
                            if reach == 0:
                                break
                            self.pub.publish(self.goal)
                            rospy.sleep(0.5)
                            reach = len(self.route)
                            tock = time.time()
                            ticktock = tock - tick
                            if ticktock >= 110:
                                break

                self.memory = 0
                viable_tp_count = 0 #Variable keeps track of viable trajectory points in a set of iterations of ns.

            if start=="start":
                try:
                    if point_ready == True:
                        TP_DS_R = [TP_DS_R[self.count - 1][0],   TP_DS_R[self.count - 1][1],   TP_DS_R[self.count - 1][2],
                                  self.goal.pose.position.x, self.goal.pose.position.y]
                    else:
                        self.listener.waitForTransform('map', 'base_footprint', rospy.Time(0), rospy.Duration(0.01))
                        (trans, rot) = self.listener.lookupTransform('map', 'base_footprint', rospy.Time(0))
                        cell_x_y = self.x_y_M_to_cell_location([trans[0], trans[1]])
                        target = self.shift([0,0,0,0,0], cell_x_y[0], cell_x_y[1], [], 10)
                        TP_DS_R = [self.x, self.y, self.z, target["arnpnt"][3], target["arnpnt"][4]]

                except:
                    self.listener.waitForTransform('map', 'base_footprint', rospy.Time(0), rospy.Duration(0.1))
                    (trans, rot) = self.listener.lookupTransform('map', 'base_footprint', rospy.Time(0))
                    cell_x_y = self.x_y_M_to_cell_location([trans[0], trans[1]])
                    target = self.shift([0, 0, 0, 0, 0], cell_x_y[0], cell_x_y[1], [], 10)
                    TP_DS_R = [self.x, self.y, self.z, target["arnpnt"][3], target["arnpnt"][4]]
                    """self.goal.pose.position.x = target["arnpnt"][3]
                    self.goal.pose.position.y = target["arnpnt"][4]
                    self.pub.publish(self.goal)
                    rospy.sleep(0.5)
                    reach = len(self.route)
                    hlfway = reach / 2
                    ptick = time.time()

                    while reach >= dist_to_goal:
                        self.pub.publish(self.goal)
                        rospy.sleep(0.5)
                        reach = len(self.route)
                        ptock = time.time()
                        pticktock = ptock - ptick
                        if pticktock > 550 and reach <= hlfway:
                            break
                        if reach <= 80:
                            tick = time.time()
                            while reach > dist_to_goal:
                                self.pub.publish(self.goal)
                                rospy.sleep(0.5)
                                reach = len(self.route)
                                tock = time.time()
                                ticktock = tock - tick
                                if ticktock >= 110:
                                    break
                    TP_DS_R = [self.x, self.y, self.z, self.goal.pose.position.x, self.goal.pose.position.y]"""

            if self.set == "NEW SET":
                self.set = ""
                self.IC_vec = [self.x, self.y, self.z, self.goal.pose.position.x , self.goal.pose.position.y]
                TP_DS_R = self.IC_vec

            prev_tp = [TP_DS_R[3],  TP_DS_R[4]]
            Tp = []
            n_TP_DS_R = TP_DS_R

            start = "start"
            for i in range(0, ns-1):
                self.stage = 1
                arnpnt_0 = arnold_RK4(A, B, C, v, n_TP_DS_R[0], n_TP_DS_R[1], n_TP_DS_R[2], n_TP_DS_R[3], n_TP_DS_R[4], dt, DS_ind)
                arnpnt = arnpnt_0
                index1 = DS_ind
                cell_x = math.ceil((arnpnt[3] - (self.map_origin_x)) / self.map_resolution)
                cell_y = math.ceil((arnpnt[4] - (self.map_origin_y)) / self.map_resolution)
                ind = self.cell_location_to_ind([cell_x, cell_y])
                if cell_x >= self.map_width or cell_y >= self.map_height or  cell_x  <= 0 or cell_y <= 0:
                    n_TP_DS_R = arnpnt
                    Tp.append([n_TP_DS_R[3], n_TP_DS_R[4]])
                    prev_tp = [n_TP_DS_R[3], n_TP_DS_R[4]]
                    TP_DS_R = np.vstack([TP_DS_R, n_TP_DS_R])
                    break

                if self.PDA[ind] == 0:
                    n_TP_DS_R = arnpnt
                    Tp.append([n_TP_DS_R[3], n_TP_DS_R[4]])
                    prev_tp = [n_TP_DS_R[3],n_TP_DS_R[4]]
                    TP_DS_R = np.vstack([TP_DS_R, n_TP_DS_R])

                elif  self.PDA[ind] != 0:
                    target1 = self.shift(arnpnt, cell_x, cell_y, prev_tp, 9)
                    if target1["cost"] >= 69: #1600 #200 FIRST
                        for i in range(0, 3):
                            if i != index1:
                                index2 = i
                                break
                        arnpnt_1 = arnold_RK4(A, B, C, v, n_TP_DS_R[0], n_TP_DS_R[1], n_TP_DS_R[2],n_TP_DS_R[3],n_TP_DS_R[4], dt, index2)
                        arnpnt = arnpnt_1
                        cell_x = math.ceil((arnpnt[3] - (self.map_origin_x)) / self.map_resolution)
                        cell_y = math.ceil((arnpnt[4] - (self.map_origin_y)) / self.map_resolution)
                        ind = self.cell_location_to_ind([cell_x, cell_y])
                        if cell_x >= self.map_width or cell_y >= self.map_height or cell_x <= 0 or cell_y <= 0:
                            n_TP_DS_R = arnpnt_0 #arnpnt_0
                            Tp.append([n_TP_DS_R[3], n_TP_DS_R[4]])
                            prev_tp = [n_TP_DS_R[3],n_TP_DS_R[4]]
                            TP_DS_R = np.vstack([TP_DS_R, n_TP_DS_R])
                            break
                        if self.PDA[ind] == 0:
                            n_TP_DS_R = arnpnt
                            Tp.append([n_TP_DS_R[3], n_TP_DS_R[4]])
                            prev_tp = [n_TP_DS_R[3],n_TP_DS_R[4]]
                            TP_DS_R = np.vstack([TP_DS_R, n_TP_DS_R])
                        elif self.PDA[ind] != 0:
                            target2 = self.shift(arnpnt, cell_x, cell_y, prev_tp, 9)  # 7
                            if target2["cost"] >= 69: #SECOND
                                for i in range(0, 3):
                                    if i != index2 and i != index1:
                                        index3 = i
                                        break
                                arnpnt = arnold_RK4(A, B, C, v, n_TP_DS_R[0], n_TP_DS_R[1],n_TP_DS_R[2],n_TP_DS_R[3],n_TP_DS_R[4],dt, index3)
                                cell_x = math.ceil((arnpnt[3] - (self.map_origin_x)) / self.map_resolution)
                                cell_y = math.ceil((arnpnt[4] - (self.map_origin_y)) / self.map_resolution)
                                ind = self.cell_location_to_ind([cell_x,  cell_y])
                                if cell_x>=self.map_width or  cell_y>=self.map_height or cell_x<=0 or  cell_y<=0:
                                    n_TP_DS_R = arnpnt_1 #arnpnt_1
                                    Tp.append([n_TP_DS_R[3], n_TP_DS_R[4]])
                                    prev_tp = [n_TP_DS_R[3],n_TP_DS_R[4]]
                                    TP_DS_R = np.vstack([TP_DS_R, n_TP_DS_R])
                                    break
                                if self.PDA[ind] == 0:
                                    n_TP_DS_R = arnpnt
                                    Tp.append([n_TP_DS_R[3], n_TP_DS_R[4]])
                                    prev_tp = [n_TP_DS_R[3],n_TP_DS_R[4]]
                                    TP_DS_R = np.vstack([TP_DS_R, n_TP_DS_R])
                                elif self.PDA[ind] != 0:
                                    minimum = min([target1["cost"], target2["cost"]])
                                    if minimum == target1["cost"]:
                                        n_TP_DS_R = target1["arnpnt"]
                                        prev_tp = [n_TP_DS_R[3],n_TP_DS_R[4]]
                                        TP_DS_R = np.vstack([TP_DS_R, n_TP_DS_R])
                                    else:
                                        n_TP_DS_R = target2["arnpnt"]
                                        prev_tp = [n_TP_DS_R[3],n_TP_DS_R[4]]
                                        TP_DS_R = np.vstack([TP_DS_R, n_TP_DS_R])
                                    Tp.append([n_TP_DS_R[3], n_TP_DS_R[4]])
                            else:
                                n_TP_DS_R = target2["arnpnt"]
                                Tp.append([n_TP_DS_R[3], n_TP_DS_R[4]])
                                prev_tp = [n_TP_DS_R[3],n_TP_DS_R[4]]
                                TP_DS_R = np.vstack([TP_DS_R, n_TP_DS_R])
                    else:
                        n_TP_DS_R = target1["arnpnt"]
                        prev_tp = [n_TP_DS_R[3],n_TP_DS_R[4]]
                        TP_DS_R = np.vstack([TP_DS_R, n_TP_DS_R])
                        Tp.append([n_TP_DS_R[3], n_TP_DS_R[4]])

            self.count = 0
            viable_tp_count = 0

            """THIS SEGMENT OF ARNOLDLOGISTIC_COVERAGE MAKES THE ROBOT MOVE FROM GOAL TO GOAL VIA THE SET OF POINTS CREATED"""
            while self.count < len(Tp):
                self.stage = 2
                no_shift=1
                if self.start_stop == "stop":
                    break

                cell_x = math.ceil((Tp[self.count][0] - (self.map_origin_x)) / self.map_resolution)
                cell_y = math.ceil((Tp[self.count][1] - (self.map_origin_y)) / self.map_resolution)


                if cell_x >= self.map_width or cell_y >= self.map_height or cell_x <= 0 or cell_y <= 0:
                    point_ready = False
                    self.count = self.count + 1
                    continue


                try:
                    if self.cost_calculator(cell_x, cell_y) >= 50 or  self.PDA[int((cell_y * self.map_width) + cell_x)] == -1 or  self.PDA[int((cell_y * self.map_width) + cell_x)] == 100:
                        self.listener.waitForTransform('map', 'base_footprint', rospy.Time(0),rospy.Duration(0.01))
                        (trans, rot) = self.listener.lookupTransform('map', 'base_footprint', rospy.Time(0))
                        cell_x_y = self.x_y_M_to_cell_location([trans[0], trans[1]])
                        target = self.shift([0,0,0,0,0], cell_x_y[0], cell_x_y[1], [], 10)

                        if target["cost"] == 10e20:
                            point_ready = False
                        else:
                            Tp[self.count][0] = target["arnpnt"][3]
                            Tp[self.count][1] = target["arnpnt"][4]
                            self.goal.pose.position.x = Tp[self.count][0]
                            self.goal.pose.position.y = Tp[self.count][1]
                            point_ready = True
                    else:
                        point_ready = True
                        self.goal.pose.position.x = Tp[self.count][0]
                        self.goal.pose.position.y = Tp[self.count][1]
                    no_shift=0

                except IndexError:
                    self.listener.waitForTransform('map', 'base_footprint', rospy.Time(0), rospy.Duration(0.01))
                    (trans, rot) = self.listener.lookupTransform('map', 'base_footprint', rospy.Time(0))
                    cell_x_y = self.x_y_M_to_cell_location([trans[0], trans[1]])
                    target = self.shift([0, 0, 0, 0, 0], cell_x_y[0], cell_x_y[1], [], 10)

                    if target["cost"] == 10e20:
                        point_ready = False
                    else:
                        Tp[self.count][0] = target["arnpnt"][3]
                        Tp[self.count][1] = target["arnpnt"][4]
                        self.goal.pose.position.x = Tp[self.count][0]
                        self.goal.pose.position.y = Tp[self.count][1]
                        point_ready = True
                    no_shift=0

                if no_shift:
                    point_ready = True
                    self.goal.pose.position.x = Tp[self.count][0]
                    self.goal.pose.position.y = Tp[self.count][1]

                if point_ready == False:
                    self.count += 1
                    continue

                if point_ready == True:
                    viable_tp_count += 1
                    self.count = self.count + 1
                    self.pub.publish(self.goal)
                    rospy.sleep(0.3)
                    reach = len(self.route)
                    hlfway = reach / 2
                    ptick = time.time()
                    while reach >= dist_to_goal:
                        if reach == 0:
                            break
                        self.pub.publish(self.goal)
                        rospy.sleep(0.3)
                        reach = len(self.route)
                        ptock = time.time()
                        pticktock = ptock - ptick
                        if pticktock > 550 and reach <= hlfway:
                            break
                        if reach <= 80:
                            tick = time.time()
                            while reach > dist_to_goal:
                                if reach == 0:
                                    break
                                self.pub.publish(self.goal)
                                rospy.sleep(0.3)
                                reach = len(self.route)
                                tock = time.time()
                                ticktock = tock - tick
                                if ticktock >= 110:
                                    break
                self.memory += self.count
                if viable_tp_count < 5:
                    bad_seed_count += 1

    def drive(self):
        rospy.Subscriber("/move_base/NavfnROS/plan", Path, self.path_watcher, queue_size=1)
        rospy.Subscriber("/zones", Float64MultiArray, self.choose_marker, queue_size=1)

        v = rospy.get_param("v")
        A = rospy.get_param("A")
        B = rospy.get_param("B")
        C = rospy.get_param("C")

        while self.start_stop != "start":
            continue

        rospy.loginfo("Chaotic coverage path planner starting .....")
        self.ArnoldLogistic_coverage(A, B, C, v)

if __name__ == "__main__":
    try:
        arny = Chaotic_system()
        rospy.Subscriber("/startup_shutdown", String, arny.call_back)
        arny.mapmaker()
        arny.drive()

    except rospy.ROSInterruptException:
        pass




