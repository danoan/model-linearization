import numpy as np

from skimage.util import img_as_bool
from skimage.morphology import binary_dilation,binary_erosion,square,disk
from skimage.draw import circle


def expand(img_mask):
    expanded_mask = np.zeros(img_mask.shape,dtype=bool)

    expanded_mask[:,:] = img_mask[:,:]

    expanded_mask[1:,:] += img_mask[:-1,:]
    expanded_mask[:-1,:] += img_mask[1:,:]

    expanded_mask[:,1:] += img_mask[:,:-1]
    expanded_mask[:,:-1] += img_mask[:,1:]

    return expanded_mask

def shrink(img_mask):
    boundary = get_boundary_mask(img_mask)[1]
    return minus(img_mask,boundary)


def identify_corners(img_mask):
    temp = np.zeros(img_mask.shape,dtype=np.int32)
    labeled_boundary = np.zeros(img_mask.shape,dtype=np.bool)


    temp[1:,:] += 1*img_mask[:-1,:] #Pull Up
    temp[:-1,:] += 2*img_mask[1:,:] #Push Down

    temp[:,1:] += 4*img_mask[:,:-1] #Pull Left
    temp[:,:-1] += 8*img_mask[:,1:] #Push Right

    temp[img_mask] = temp[img_mask]
    temp[np.bitwise_not(img_mask)]=0

    labeled_boundary[temp==5] = True
    labeled_boundary[temp==9] = True
    labeled_boundary[temp==6] = True
    labeled_boundary[temp==10] = True


    return labeled_boundary



def euclidean_disk(x,y,r,shape):
    squares_mask = (r+1)**2*np.ones( shape )

    squares = [ i**2 for i in range(r+1) ]
    for i in range(r+1):
        for j in range(r+1):
            squares_mask[y+i,x+j] = squares[i] + squares[j]    

    for i in range(r+1):
        for j in range(r+1):
            squares_mask[y-i,x+j] = squares[i] + squares[j]    
            

    for i in range(r+1):
        for j in range(r+1):
            squares_mask[y-i,x-j] = squares[i] + squares[j]                
    

    for i in range(r+1):
        for j in range(r+1):
            squares_mask[y+i,x-j] = squares[i] + squares[j]                    


    squares_mask[squares_mask<=r**2] =1
    squares_mask[squares_mask>1] = 0
    
    return squares_mask==1


def neighborhood(p,h,w):
    y,x = p
    N = [ check_bounds( (y+1,x),h,w ),
          check_bounds( (y-1,x),h,w ),
          check_bounds( (y,x+1),h,w ),
          check_bounds( (y,x-1),h,w ) ]
    return N

def check_bounds(p,h,w):
    y,x = p
    y = h-1 if y>=h else y
    y = 0 if y<0 else y
    x = w-1 if x>=w else x
    x = 0 if x<0 else x    

    return ( y,x )

def get_boundary_mask(img):
    '''
        Return the boundary of a digital image.

        img: must be a binary image. 
             1 - Foreground
             0 - Background

        return: Boolean image. Values of 1 indicates points of the boundary.
    '''
    if (img>1).any():
        raise Exception("Image is not binary.")

    imbool = img_as_bool(img)
    h,w = imbool.shape


    
    temp = np.array(imbool,dtype=int)
    temp[imbool==0] = -4

    temp[1:,:]  +=  imbool[:-1,:]
    temp[:-1,:] +=  imbool[1:,:]

    temp[:,1:]  +=  imbool[:,:-1]
    temp[:,:-1] +=  imbool[:,1:]    

    boundary_mask = np.zeros(imbool.shape,dtype=bool)
    boundary_mask[ temp>=2 ] = temp[ temp>=2 ] < 5


    return (h,w),boundary_mask

def get_boundary_mask_D(img):
    '''
        Return the boundary of a digital image.

        img: must be a binary image. 
             1 - Foreground
             0 - Background

        return: Boolean image. Values of 1 indicates points of the boundary.
    '''
    if (img>1).any():
        raise Exception("Image is not binary.")

    imbool = img_as_bool(img)
    h,w = imbool.shape


    boundary_mask = minus(img,binary_erosion(img,square(3)))
    # boundary_mask = imbool

    return (h,w),boundary_mask    

def get_boundary_mask_F(img):
    '''
        Return the boundary of a digital image.

        img: must be a binary image. 
             1 - Foreground
             0 - Background

        return: Boolean image. Values of 1 indicates points of the boundary.
    '''
    if (img>1).any():
        raise Exception("Image is not binary.")

    imbool = img_as_bool(img)
    h,w = imbool.shape

    boundary_mask = binary_dilation(img,square(3))

    return (h,w),boundary_mask     

def binary_erosion_region_and_boundary_masks(img,radius,selem):
    '''
        img: must be a binary image. 
             1 - Foreground
             0 - Background

        return: (Boolean image,Boolean image). Values of 1 indicates points of the region/boundary.
    '''
    (h,w),border = get_boundary_mask(img)

    if radius==0:
        trust_region_mask = img
    else:
        trust_region_mask = binary_erosion(img,selem(radius))
        
    (a,b),trust_boundary_mask = get_boundary_mask( trust_region_mask )

    return (trust_region_mask,trust_boundary_mask)


def binary_dilation_region_and_boundary_masks(img,radius,selem):
    '''
        img: must be a binary image. 
             1 - Foreground
             0 - Background

        return: (Boolean image,Boolean image). Values of 1 indicates points of the region/boundary.
    '''    
    (h,w),border = get_boundary_mask(img)

    extended_region_mask = binary_dilation(img,selem(radius))
    (a,b),extended_boundary_mask = get_boundary_mask( extended_region_mask )    

    return (extended_region_mask,extended_boundary_mask)


def get_boundaries_and_regions_masks(img,erosion_radius,dilation_radius,selem):
    '''
        img: must be a binary image. 
             1 - Foreground
             0 - Background

        return: (Boolean image,Boolean image). Values of 1 indicates points of the region/boundary.
    '''        
    trust_region_mask,trust_boundary_mask = get_trust_region_and_boundary_masks(img,erosion_radius,selem)
    middle_region_mask,middle_boundary_mask = get_middle_region_and_boundary_masks(img)
    extended_region_mask,extended_boundary_mask = get_extended_region_and_boundary_masks(img,dilation_radius,selem)
    
    return (trust_region_mask,middle_region_mask,extended_region_mask,trust_boundary_mask,middle_boundary_mask,extended_boundary_mask)


def generate_disk_region_mask(cx,cy,r,shape=None):
    '''
        return: Boolean image. Values of 1 indicates points in the disk.
    '''

    if shape is None:
        shape = (4*r,4*r)
    
    disk_mask = np.zeros(shape,dtype=bool)
    ccy,ccx = circle(cy,cx,r,shape)
    disk_mask[ccy,ccx] = True

    # disk_mask = euclidean_disk(cx,cy,r,shape)

    return disk_mask


def generate_disk_region_mask_automatic_center(radius,shape=None):
    return generate_disk_region_mask(2*radius,2*radius,radius,shape)  

def compute_disk_area(radius):
    D=generate_disk_region_mask_automatic_center(radius)
    return len(D[D])


def intersect(region_mask_1,region_mask_2):    
    return np.bitwise_and(region_mask_1,region_mask_2)

def minus(region_mask_1,region_mask_2):
    return np.bitwise_and( region_mask_1, ~region_mask_2 )


def build_coordinates_matrix(h,w):    
    coord_matrix = []
    for i in range(h):
        coord_matrix.append([])
        for j in range(w):
            coord_matrix[-1].append( (i,j) )
    coord_matrix = np.array( coord_matrix, dtype=[('y','>i4'),('x','>i4')] )

    return coord_matrix

