"""
Script that uses calibration methods to find the instrisic and extrinsic
parameters of the cameras
"""
import pickle
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import scipy.optimize
import matplotlib.pyplot as plt
import pickle
from pathlib import Path


def makerotation(rx,ry,rz):
    """
    Generate a rotation matrix    

    Parameters
    ----------
    rx,ry,rz : floats
        Amount to rotate around x, y and z axes in degrees

    Returns
    -------
    R : 2D numpy.array (dtype=float)
        Rotation matrix of shape (3,3)
    """
    rx = np.pi*rx/180.0
    ry = np.pi*ry/180.0
    rz = np.pi*rz/180.0

    Rx = np.array([[1,0,0],[0,np.cos(rx),-np.sin(rx)],[0,np.sin(rx),np.cos(rx)]])
    Ry = np.array([[np.cos(ry),0,-np.sin(ry)],[0,1,0],[np.sin(ry),0,np.cos(ry)]])
    Rz = np.array([[np.cos(rz),-np.sin(rz),0],[np.sin(rz),np.cos(rz),0],[0,0,1]])
    R = (Rz @ Ry @ Rx)
    
    return R 


def internal():
    """
    Finds the instrisic parameters of the camera. Taken from calibrate.py from assignment 3
    params:
        None
    returns: 
        Dictionary of the intrinsic parameters
    """
    # Check if pickled camera file already exists
    if not Path('pickle/calibration.pkl').is_file():
        # file names, modify as necessary
        calibimgfiles = './scans/calib_jpg_u/*.jpg'
        resultfile = 'pickle/calibration.pkl'

        # checkerboard coordinates in 3D
        objp = np.zeros((6*8,3), np.float32)
        objp[:,:2] = 2.8*np.mgrid[0:8, 0:6].T.reshape(-1,2)

        # arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.

        # Make a list of calibration images
        images = glob.glob(calibimgfiles)

        # Step through the list and search for chessboard corners
        for idx, fname in enumerate(images):
            img = cv2.imread(fname)
            img_size = (img.shape[1], img.shape[0])
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (8,6), None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

                # Display image with the corners overlayed
                cv2.drawChessboardCorners(img, (8,6), corners, ret)
                cv2.imshow('img', img)
                cv2.waitKey(500)
        cv2.destroyAllWindows()

        # now perform the calibration
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)

        print("Estimated camera intrinsic parameter matrix K")
        print(K)
        print("Estimated radial distortion coefficients")
        print(dist)

        print("Individual intrinsic parameters")
        print("fx = ",K[0][0])
        print("fy = ",K[1][1])
        print("cx = ",K[0][2])
        print("cy = ",K[1][2])


        # save the results out to a file for later use
        calib = {}
        calib["fx"] = K[0][0]
        calib["fy"] = K[1][1]
        calib["cx"] = K[0][2]
        calib["cy"] = K[1][2]
        calib["dist"] = dist

        fid = open(resultfile, "wb" ) 
        pickle.dump(calib,fid)
        fid.close()
    else:
        print(f'Camera already calibrated. Use the pickled file instead')


def external():
    """
    Uses a dictionary of the camera parameters and least squares optimization
    to find the extrinisic parameters of the camera. 
    params:
        params (dict): dictionary of instrinsic camera paramters
    returns:
    """
    # Check if pickled camera already exists
    if not Path('pickle/camera.pkl').is_file():
        # load in the intrinsic camera parameters from 'calibration.pickle'
        params = pickle.load(open('pickle/calibration.pkl', 'rb'))
        
        # Initialize intrinsic camera parameters
        camL = Camera(
            f=params['fx'],c=np.array([[params['cx'], params['cy']]]).T,
            t=np.array([[0,0,0]]).T,
            R=makerotation(0,0,0)
        )
        
        camR = Camera(
            f=params['fy'],c=np.array([[params['cx'], params['cy']]]).T,
            t=np.array([[0,0,0]]).T,
            R=makerotation(0,0,0)
        )

        # load in the left and right images and find the coordinates of
        # the chessboard corners using OpenCV
        imgL = plt.imread('./scans/calib_jpg_u/frame_C0_01.jpg')
        ret, cornersL = cv2.findChessboardCorners(imgL, (8,6), None)
        pts2L = cornersL.squeeze().T

        imgR = plt.imread('./scans/calib_jpg_u/frame_C1_01.jpg')
        ret, cornersR = cv2.findChessboardCorners(imgR, (8,6), None)
        pts2R = cornersR.squeeze().T

        # generate the known 3D point coordinates of points on the checkerboard in cm
        pts3 = np.zeros((3,6*8))
        yy,xx = np.meshgrid(np.arange(8),np.arange(6))
        pts3[0,:] = 2.8*xx.reshape(1,-1)
        pts3[1,:] = 2.8*yy.reshape(1,-1)

        # Use calibratePose helper function to find and update extrinsic parameters
        params_initL = np.array([0,0,0,0,0,2]) 
        params_initR = np.array([0,0,0,0,0,2]) 
        camL = calibratePose(pts3,pts2L,camL,params_initL)
        camR = calibratePose(pts3,pts2R,camR,params_initR)
        
        fid = open('pickle/camera.pkl', "wb" ) 
        pickle.dump((camL, camR),fid)
        fid.close()
        print('Done!')
    else:
        print('Camera parameters have been pickled already. Just load from file!')

def residuals(pts3,pts2,cam,params):
    """
    Compute the difference between the projection of 3D points by the camera
    with the given parameters and the observed 2D locations

    Parameters
    ----------
    pts3 : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (3,N)

    pts2 : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (2,N)

    params : 1D numpy.array (dtype=float)
        Camera parameters we are optimizing over stored in a vector

    Returns
    -------
    residual : 1D numpy.array (dtype=float)
        Vector of residual 2D projection errors of size 2*N
        
    """

    cam.update_extrinsics(params)
    residual = pts2 - cam.project(pts3)
    
    return residual.flatten()

def calibratePose(pts3,pts2,cam_init,params_init):
    """
    Calibrate the provided camera by updating R,t so that pts3 projects
    as close as possible to pts2

    Parameters
    ----------
    pts3 : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (3,N)

    pts2 : 2D numpy.array (dtype=float)
        Coordinates of N points stored in a array of shape (2,N)

    cam : Camera
        Initial estimate of camera

    Returns
    -------
    cam_opt : Camera
        Refined estimate of camera with updated R,t parameters
        
    """
    # define our error function
    efun = lambda params: residuals(pts3,pts2,cam_init,params)        
    popt,_ = scipy.optimize.leastsq(efun,params_init)
    cam_init.update_extrinsics(popt)


class Camera:
    """
    A simple data structure describing camera parameters 
    
    The parameters describing the camera
    cam.f : float   --- camera focal length (in units of pixels)
    cam.c : 2x1 vector  --- offset of principle point
    cam.R : 3x3 matrix --- camera rotation
    cam.t : 3x1 vector --- camera translation 

    
    """    
    def __init__(self,f,c,R,t):
        self.f = f
        self.c = c
        self.R = R
        self.t = t

    def __str__(self):
        return f'Camera : \n f={self.f} \n c={self.c.T} \n R={self.R} \n t = {self.t.T}'
    
    def project(self,pts3):
        """
        Project the given 3D points in world coordinates into the specified camera    

        Parameters
        ----------
        pts3 : 2D numpy.array (dtype=float)
            Coordinates of N points stored in a array of shape (3,N)

        Returns
        -------
        pts2 : 2D numpy.array (dtype=float)
            Image coordinates of N points stored in an array of shape (2,N)

        """
        assert(pts3.shape[0]==3)

        # get point location relative to camera
        pcam = self.R.transpose() @ (pts3 - self.t)
         
        # project
        p = self.f * (pcam / pcam[2,:])
        
        # offset principal point
        pts2 = p[0:2,:] + self.c
        
        assert(pts2.shape[1]==pts3.shape[1])
        assert(pts2.shape[0]==2)
    
        return pts2
 
    def update_extrinsics(self,params):
        """
        Given a vector of extrinsic parameters, update the camera
        to use the provided parameters.
  
        Parameters
        ----------
        params : 1D numpy.array (dtype=float)
            Camera parameters we are optimizing over stored in a vector
            params[0:2] are the rotation angles, params[2:5] are the translation

        """
        self.R = makerotation(params[0],params[1],params[2])
        self.t = np.array([[params[3]],[params[4]],[params[5]]])

if __name__ == "__main__":
    internal()
    external()