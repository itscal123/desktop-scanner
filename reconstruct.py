"""
Take a set of color images from a pair of cameras to generate a 3D point cloud
"""


def decode(imprefix,start,threshold):
   """
   Given a sequence of images of a scene showing projected 10 bit gray code, 
   decode the binary sequence into a decimal value in (0,1023) for each pixel.
   Mark those pixels whose code is likely to be incorrect based on the user 
   provided threshold.  Images are assumed to be named "imageprefixN.png" where
   N is a 2 digit index (e.g., "img00.png,img01.png,img02.png...")

   Parameters
   ----------
   imprefix : str
      Image name prefix
   
   start : int
      Starting index
      
   threshold : float
      Threshold to determine if a bit is decodeable
      
   Returns
   -------
   code : 2D numpy.array (dtype=float)
      Array the same size as input images with entries in (0..1023)
      
   mask : 2D numpy.array (dtype=bool)
      Array indicating which pixels were correctly decoded based on the threshold
   
   """
   
   # we will assume a 10 bit code
   nbits = 10
      
   # Empty stack to track all the binary images
   stack = None

   # Initialize code
   code = None

   # Initialize mask
   mask = None

   # Load images, note each image has an inverse, so a total of 20 images are processed
   for i in range(start, start+20, 2):
      # Load image
      base = plt.imread(f'{imprefix}{i:02}.png')
      inverse = plt.imread(f'{imprefix}{i+1:02}.png')

      # Convert to float data type and scale to [0..1] if necessary
      if (base.dtype == np.uint8):
         base = base.astype(float) / 256
         inverse = inverse.astype(float) / 256

      # Convert RGB image to grayscale if necessary
      if len(base.shape) == 3:
         base = np.mean(base, axis=2)
         inverse = np.mean(inverse, axis=2)

      # Initialize code and mask on first iteration
      if code is None and mask is None:
         code = np.zeros(base.shape, dtype=int)
         mask = np.ones(base.shape, dtype=bool)

      # Mark corresponding position in mask for every undecodable bit 
      mask[np.absolute(base - inverse) < threshold] = 0
       
      if stack is None:    # initialize on first iteration
         stack = base > inverse

      else:                # otherwise, add to stack
         stack = np.concatenate((stack, base > inverse), axis=0)

   # Reshape the stack into 3D array
   stack = stack.reshape(10, base.shape[0], base.shape[1])

   # Stack along the last dimension
   stack = np.stack(stack, axis=-1)

   # Convert the grayscale bit into BCD, then convert BCD to decimal
   for bit in range(10):
      # First bits are the same
      if bit == 0:
         prev = stack[:,:,bit]
         curr = stack[:,:,bit]
      # BCD bit = previous BCD bit XOR current grayscale bit
      else:
         curr = np.logical_xor(stack[:,:,bit], prev)
         prev = curr
      
      # Add the decimal conversion the code array
      code += curr * 2 ** (9-bit)

   return code, mask


def reconstruct(imprefixL,imprefixR,threshold,camL,camR):
    """
    Performing matching and triangulation of points on the surface using structured
    illumination. This function decodes the binary graycode patterns, matches 
    pixels with corresponding codes, and triangulates the result.
    
    The returned arrays include 2D and 3D coordinates of only those pixels which
    were triangulated where pts3[:,i] is the 3D coordinte produced by triangulating
    pts2L[:,i] and pts2R[:,i]

    Parameters
    ----------
    imprefixL, imprefixR : str
        Image prefixes for the coded images from the left and right camera
        
    threshold : float
        Threshold to determine if a bit is decodeable
   
    camL,camR : Camera
        Calibration info for the left and right cameras
        
    Returns
    -------
    pts2L,pts2R : 2D numpy.array (dtype=float)
        The 2D pixel coordinates of the matched pixels in the left and right
        image stored in arrays of shape 2xN
        
    pts3 : 2D numpy.array (dtype=float)
        Triangulated 3D coordinates stored in an array of shape 3xN
        
    """

    # Decode the H and V coordinates for the two views
    V_left_code, V_left_mask = decode(imprefixL, 0, threshold)
    H_left_code, H_left_mask = decode(imprefixL, 20, threshold)

    V_right_code, V_right_mask = decode(imprefixR, 0, threshold)
    H_right_code, H_right_mask = decode(imprefixR, 20, threshold)

    # Set the height and width of the images
    h, w = V_left_code.shape

    # Construct the combined 20 bit code C = H + 1024*V and mask for each view
    CL = (H_left_code * H_left_mask) + 1024 * (V_left_code * V_left_mask)
    CR = (H_right_code * H_right_mask) + 1024 * (V_right_code * V_right_mask)
    
    # Find the indices of pixels in the left and right code image that 
    # have matching codes. If there are multiple matches, just
    # choose one arbitrarily.
    _, matchL, matchR = np.intersect1d(CL, CR, return_indices=True)
    
    # Let CL and CR be the flattened arrays of codes for the left and right view
    # Suppose you have computed arrays of indices matchL and matchR so that 
    # CL[matchL[i]] == CR[matchR[i]] for all i.  The code below gives one approach
    # to generating the corresponding pixel coordinates for the matched pixels.
    
    xx,yy = np.meshgrid(range(w),range(h))
    xx = np.reshape(xx,(-1,1))
    yy = np.reshape(yy,(-1,1))
    pts2R = np.concatenate((xx[matchR].T,yy[matchR].T),axis=0)
    pts2L = np.concatenate((xx[matchL].T,yy[matchL].T),axis=0)

    # Now triangulate the points
    pts3 = triangulate(pts2L,camL,pts2R,camR)
    
    
    return pts2L,pts2R,pts3