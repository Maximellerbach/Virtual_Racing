import math
import random as rn

import cv2
import numpy as np
import scipy as sc
from scipy import interpolate
from scipy.spatial import ConvexHull

def get_circuit_bounds(pts):
    pts = np.array(pts)
    return (int(np.max(pts[:, -1])), int(np.max(pts[:, 0])), 3)

def dist_3D(p, p2):
    return math.sqrt((p[0]-p2[0])**2 + (p[1]-p2[1])**2 + (p[2]-p2[2])**2)

# defining track width, height and z-axis height
WIDTH = 250
HEIGHT = 250
DEPTH = 10

# Boundaries for the numbers of points that will be randomly 
# generated to define the initial polygon used to build the track
MIN_POINTS = 15
MAX_POINTS = 20

SPLINE_POINTS = 100

# Margin between screen limits and any of the points that shape the
# initial polygon
MARGIN = 25
# minimum distance between points that form the track skeleton
MIN_DISTANCE = 40
# Maximum midpoint displacement for points placed after obtaining the initial polygon
MAX_DISPLACEMENT = 10

# Track difficulty
DIFFICULTY = 0.1
# min distance between two points that are part of the track skeleton
DISTANCE_BETWEEN_POINTS = 40
# Maximum corner allowed angle
MAX_ANGLE = 90


class TrackGenerator():
    def __init__(self):
        self._track_points = []
        self._checkpoints = []

    # track generation methods
    def generate_track(self):
        # generate the track
        self._points = self.random_points()
        pts2D = np.array([self._points[:, 0], self._points[:, -1]]) 
        pts2D = np.transpose(pts2D, (1, 0))

        self._hull = ConvexHull(pts2D)
        
        self._track_points = self.get_track_points_from_hull(self._hull, self._points)
        # self.draw_pts(self._track_points, (HEIGHT, WIDTH, 3), fill=-1)
        
        self._track_points = self.shape_track(self._track_points)
        # self.draw_pts(self._track_points, (HEIGHT, WIDTH, 3), fill=-1)

        self._track_points = self.smooth_track(self._track_points, s=0)
        # self.draw_pts(self._track_points, (HEIGHT, WIDTH, 3), fill=-1)

        self._track_points = self.check_angles(self._track_points)
        self.draw_pts(self._track_points, (HEIGHT, WIDTH, 3), fill=-1)

        is_ring, is_valid = self.check_is_valid(self._track_points)

        if is_valid:
            self.modify_track()

        
    def random_points(self, min=MIN_POINTS, max=MAX_POINTS, margin=MARGIN, min_distance=MIN_DISTANCE):
        point_count = rn.randrange(min, max+1, 1)
        points = []
        for i in range(point_count):
            x = rn.randrange(margin, WIDTH - margin + 1, 1)
            z = rn.uniform(0, 1)*DEPTH
            y = rn.randrange(margin, HEIGHT -margin + 1, 1)
            distances = list( filter(lambda x: x < min_distance, [dist_3D(p, (x, z, y)) for p in points]) )
            if len(distances) == 0:
                points.append((x, z, y))
        return np.array(points)

    def get_track_points_from_hull(self, hull, points):
        # get the original points from the random 
        # set that will be used as the track starting shape
        pts = [np.array(points[v]) for v in hull.vertices]
        pts.append(pts[0])
        return np.array(pts)

    def make_rand_vector(self, dims):
        vec = [rn.gauss(0, 1) for i in range(dims)]
        mag = sum(x**2 for x in vec) ** .5
        return [x/mag for x in vec]

    def shape_track(self, track_points, difficulty=DIFFICULTY, max_displacement=MAX_DISPLACEMENT, margin=MARGIN):
        def process_diff(now, prev):
            diff = np.average(np.array(prev)-np.array(now))
            return diff

        def get_into_bounds(track_set):
            # push any point outside screen limits back again
            pt_set = []
            for point in track_set:
                if point[0] < margin:
                    point[0] = margin + np.random.randint(0, max_displacement)
                elif point[0] > (WIDTH - margin):
                    point[0] = WIDTH - margin - np.random.randint(0, max_displacement)

                # if point[1] < 0: # not used for the moment as I rescale the Z axis when I save it
                #     point[1] = 0
                # elif point[1] > DEPTH:
                #     point[1] = DEPTH

                if point[-1] < margin:
                    point[-1] = margin + np.random.randint(0, max_displacement)
                elif point[-1] > HEIGHT - margin:
                    point[-1] = HEIGHT - margin - np.random.randint(0, max_displacement)
                    
                pt_set.append(point)
            return pt_set

        new_points_factor = 2 # give here an int
        track_set = [[0,0,0] for i in range(int(len(track_points)*new_points_factor))]

        assert len(track_points) != len(track_set) 
        
        for i in range(len(track_points)):
            displacement = math.pow(rn.random(), difficulty) * max_displacement
            disp = [displacement * i for i in self.make_rand_vector(3)]
            int_times_fact = int(i*new_points_factor)
            track_set[int_times_fact] = track_points[i]
            track_set[int_times_fact + 1][0] = (track_points[i][0] + track_points[(i+1)%len(track_points)][0]) / 2 + disp[0]
            track_set[int_times_fact + 1][1] = (track_points[i][1] + track_points[(i+1)%len(track_points)][1]) / 2 + disp[1]
            track_set[int_times_fact + 1][-1] = (track_points[i][-1] + track_points[(i+1)%len(track_points)][-1]) / 2 + disp[-1]

        previous = np.array(track_set)
        momentum = 10
        prev_diff = [1]*momentum
        max_it = 100
        it = 0 
        while(np.average(np.array(prev_diff)[-momentum:])>0.0 and it<max_it): # not really elegant but works
            track_set = self.fix_angles(track_set)
            track_set = self.push_points_apart(track_set)
            track_set = get_into_bounds(track_set)

            prev_diff.append(process_diff(track_set, previous))

            previous = np.array(track_set)
            it += 1

        final_set = get_into_bounds(track_set)
        return final_set

    def push_points_apart(self, points, distance=DISTANCE_BETWEEN_POINTS):
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                p_distance =  math.sqrt((points[i][0]-points[j][0])**2 + (points[i][-1]-points[j][-1])**2)
                dx, dz, dy = [0]*3
                if p_distance < distance:
                    dx = points[j][0] - points[i][0]  
                    dz = points[j][1] - points[i][1]
                    dy = points[j][-1] - points[i][-1]
                    dl = math.sqrt(dx**2 + dy**2 + dz**2)  
                    dx /= dl
                    dz /= dl
                    dy /= dl
                    dif = distance - dl
                    dx *= dif
                    dz *= dif
                    dy *= dif
                    points[j][0] = points[j][0] + dx
                    points[j][1] = points[j][1] + dz/2
                    points[j][-1] = points[j][-1] + dy
                    points[i][0] = points[i][0] - dx
                    points[i][1] = points[i][1] - dz/2
                    points[i][-1] = points[i][-1] - dy
                
        return points

    def fix_angles(self, points, max_angle=MAX_ANGLE):
        for i in range(len(points)):
            if i > 0:
                prev_point = i - 1
            else:
                prev_point = len(points)-1
            next_point = (i+1) % len(points)
            px = points[i][0] - points[prev_point][0]
            pz = points[i][1] - points[prev_point][1]
            py = points[i][-1] - points[prev_point][-1]
            pl = math.sqrt(px**2 + py**2 + pz**2)
            px /= pl
            py /= pl
            nx = -(points[i][0] - points[next_point][0])
            nz = -(points[i][1] - points[next_point][1])
            ny = -(points[i][-1] - points[next_point][-1])
            nl = math.sqrt(nx**2 + ny**2 + nz**2)
            nx /= nl
            ny /= nl

            a = math.atan2(px * ny - py * nx, px * nx + py * ny)
            if (math.degrees(abs(a)) <= max_angle):
                continue
            

            diff = math.radians(max_angle * math.copysign(1,a)) - a
            c = math.cos(diff)
            s = math.sin(diff)
                
            new_x = (nx * c - ny * s) * nl
            new_y = (nx * s + ny * c) * nl
            points[next_point][0] = points[i][0] + new_x
            points[next_point][-1] = points[i][-1] + new_y
            points[i][1] = (points[prev_point][1]*0.4+points[i][1]*0.2+points[next_point][1]*0.4)

        return points

    def smooth_track(self, track_points, s=0, n_spline=SPLINE_POINTS):
        
        x = np.array([p[0] for p in track_points])
        z = np.array([p[1] for p in track_points])
        y = np.array([p[-1] for p in track_points])

        # fit splines to x=f(u) and y=g(u), treating both as periodic.
        tck, u = interpolate.splprep([x, z, y], s=s, per=True, quiet=2)

        # evaluate the spline fits for # points evenly spaced distance values
        xi, zi, yi = interpolate.splev(np.linspace(0, 1, n_spline), tck)
        return [(xi[i], zi[i], yi[i]) for i in range(len(xi))]
        
    def check_angles(self, points, max_angle=10):
        def getAngle(a, b, c):
            ang = math.degrees(math.atan2(c[2]-b[2], c[0]-b[0]) - math.atan2(a[2]-b[2], a[0]-b[0]))
            return ang + 360 if ang < 0 else ang

        n_points = 1
        while(n_points != 0):
            n_points = 0
            prev_angle = getAngle(points[0], points[1], points[2])

            i = 0
            while(i < len(points)-1):

                angle = getAngle(points[i-1], points[i], points[i+1])
                delta_angle = angle-prev_angle

                if abs(delta_angle)> max_angle:
                    n_points += 1
                    del points[i]
                    angle = getAngle(points[i-2], points[i-1], points[i])

                prev_angle = angle
                i += 1

            points = self.smooth_track(points, s=0)
        return points

    def check_is_valid(self, points): # check whether the road is crossing at somepoint
        from shapely.geometry import LineString # module to check wether two segments are crossing
        is_ring = True
        is_valid = True

        pts_len = len(points)
        i = 0
        while(i < pts_len-1):
            couple1 = LineString([(points[i][0], points[i][-1]), (points[i+1][0], points[i+1][-1])])
            for j in range(i+2, pts_len-1):
                couple2 = LineString([(points[j][0], points[j][-1]), (points[j+1][0], points[j+1][-1])])

                inters = couple1.intersects(couple2)
                if inters:
                    is_ring = False
                    zi = points[i][1]
                    zj = points[j][1]
                    if abs(zi-zj) <= 4:
                        is_valid = False
                    break
            i += 1

        return is_ring, is_valid

    def modify_track(self):
        def rescale(points, max_d=DEPTH, max_w=200, max_h=200, border_off=25, eps=1E-10):
            points = np.array(points)

            x_axis = points[:, 0]
            x_axis = x_axis-np.min(x_axis)
            points[:, 0] = border_off+(x_axis/(np.max(x_axis)+eps)*max_w)

            y_axis = points[:, -1]
            y_axis = y_axis-np.min(y_axis)
            points[:, -1] = border_off+(y_axis/(np.max(y_axis)+eps)*max_h)


            z_axis = points[:, 1]
            z_axis = z_axis-np.min(z_axis)
            points[:, 1] = (z_axis/(np.max(z_axis)+eps)*max_d)
            return points, (max_w+border_off*2, max_h+border_off*2, 3)


        self._track_points, shape = rescale(self._track_points)

        screen = np.zeros(shape)
        screen = self.draw_pts(self._track_points, shape, fill=-1, show=False)
        cv2.imshow("track", screen)
        key = chr(cv2.waitKey(0))
        
        if key == "a": # saving/modify process
            to_save_points, start_index = self.select_start_point(self._track_points, shape)
            to_save_points = self.rotate_circuit(to_save_points, start_index, shape)
            self.save(to_save_points)
    
    def save(self, pts, x_offset=47.71, y_offset=0.6, z_offset=49.71787562201313, fact=1, save_file='C:\\Users\\maxim\\GITHUB\\sdsandbox\\sdsim\\Assets\\Resources\\track.txt'): # TODO:
        import os
        try:
            os.remove(save_file)
        except:
            pass
        
        pts.append(pts[0]) # close the loop

        pts = np.array(pts)*fact
        x_auto_off, z_auto_off, y_auto_off = -pts[0]
        offset_array = np.array((x_auto_off, z_auto_off, y_auto_off))
        for pt in pts:
            pt += offset_array
            if pt[1] < 0:
                pt[1] = 0

        if pts[1][-1] < 0.0: # check if the circuit is in the wrong "way"
            pts[:, 0] = -pts[:, 0]
            pts[:, -1] = -pts[:, -1]
        

        circuit_coords = open(save_file, 'w')
        for pt in pts:
            circuit_coords.write(str(pt[0]+x_offset)+','+str(pt[1]+y_offset)+','+str(pt[-1]+z_offset)+'\n')
        circuit_coords.close()

    def draw_pts(self, points, shape, fill=1, show=True, name="track", eps=1E-10):
        def get_Z(pt):
            return pt[1]
        points = sorted(points, key=get_Z)
        max_d = np.amax(np.array(points)[:, 1])

        screen = np.ones(shape)
        for i, point in enumerate(points):
            cv2.circle(screen, (int(point[0]), int(point[-1])), 8, (1, point[1]/(max_d+eps), 0), thickness=fill)

        if show:
            cv2.imshow(name, screen)
            cv2.waitKey(0)
        else:
            return screen


    def select_start_point(self, points, shape):
        start_index = 0
        key = ""

        while(key != "a"):
            screen = self.draw_pts(points, shape, fill=-1, show=False)
            cv2.circle(screen, (int(points[start_index][0]), int(points[start_index][-1])), 10, (0, 0, 1), thickness=3)

            cv2.imshow("track", screen)
            key = chr(cv2.waitKey(0))

            if key == "p": # plus
                start_index = int(start_index+1)
            elif key == "m": # minus
                start_index = int(start_index-1)

            start_index = start_index%len(points)

        not_empty = [i for i in (points[start_index:], points[:start_index]) if len(i)!=0]
        points = list(np.concatenate(not_empty, axis=0))
        del points[-start_index]
        return points, start_index # start_index is now 0, this is more of an offset value

    def rotate_circuit(self, points, start_index, shape):
        def rotate(origin, point, angle):
            """
            Rotate a point counterclockwise by a given angle around a given origin.

            The angle should be given in radians.
            """
            tmp_angle = math.radians(angle)
            ox, oy = origin
            px, pz, py = point

            qx = ox + math.cos(tmp_angle) * (px - ox) - math.sin(tmp_angle) * (py - oy)
            qy = oy + math.sin(tmp_angle) * (px - ox) + math.cos(tmp_angle) * (py - oy)
            return qx, pz, qy

        angle = 0
        key = ""
        mid_point = (shape[0]//2, shape[1]//2)

        new_points = points
        while(key != "a"):
            screen = self.draw_pts(new_points, shape, fill=-1, show=False)

            st = new_points[0]
            cv2.line(screen, (int(st[0]-10), int(st[-1])), (int(st[0]+10), int(st[-1])), (0, 0, 1), thickness=2)
            cv2.imshow("track", screen)
            key = chr(cv2.waitKey(0))

            if key == "p": # plus
                angle = int(angle+1)
            elif key == "m": # minus
                angle = int(angle-1)

            angle = angle%360
            new_points = [rotate(mid_point, pt, angle) for pt in points]

        return new_points
            
if __name__ == "__main__":
    t = TrackGenerator()
    while(1):
        # try:
        #     t.generate_track()
        # except:
        #     print("unable to create track")
        t.generate_track()