#coding:utf-8

import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.util import img_as_bool

import tools as MT

np.set_printoptions(threshold=1000000000,linewidth=100000000)

def domain_iterator(domain):
    lb = domain[0]
    ub = domain[1]

    for x in range(lb[0],ub[0]+1):
        for y in range(lb[1],ub[1]+1):
            yield (x,y)

def domain_dimensions(domain):
    w = domain[1][0] - domain[0][0] + 1
    h = domain[1][1] - domain[0][1] + 1

    return (w,h)

def load_image(img_path,as_bool=False,gray=False):
    img = imread(img_path,as_gray=gray)
    if as_bool:
        return img_as_bool(img)
    else:
        return img

def mask_from_image(image_mask):
    mask = np.zeros(image_mask.shape,dtype=np.bool)
    mask[image_mask] = True

    return mask

def mask_from_image_path(image_path):
    img = load_image(image_path,as_bool=True,gray=True)
    return mask_from_image(img)

def input():    
    input_image_path = "images/single_square/original.pgm"
    input_image = load_image(input_image_path,as_bool=False)

    radius = 3.0
    Q = (np.pi*radius**2)/2.0
    rows,cols = input_image.shape
    domain = ( (0,0), (cols-1,rows-1) )    


    trust_frg_mask_path = "images/single_square/trust_frg.pgm"
    trust_frg_mask = mask_from_image_path(trust_frg_mask_path)

    application_mask_path = "images/single_square/application.pgm"
    application_mask = mask_from_image_path(application_mask_path)

    optimization_mask_path = "images/single_square/optimization.pgm"
    optimization_mask = mask_from_image_path(optimization_mask_path)


    return {"radius":radius,
            "Q":Q,
            "domain": domain,
            "input_image":input_image,
            "trust_frg_mask":trust_frg_mask,
            "application_mask":application_mask,
            "optimization_mask":optimization_mask}


class VariableMap:
    def __init__(self,initial_index=0):
        self.initial_index = initial_index
        self.curr_index = initial_index

        self.point_to_index = {}
        self.index_to_point = {}
        

    def domain(self):
        return (self.initial_index,self.curr_index)

    def update(self,point,inc_index=True):
        self.point_to_index[point] = self.curr_index
        self.index_to_point[self.curr_index] = point

        if inc_index:
            self.curr_index+=1

    def distinct_count(self):
        return len(self.index_to_point.keys())

    def __getitem__(self,key):
        if type(key) is int:
            return self.index_to_point[key]
        else:
            return self.point_to_index[key]


class ModelMap:
    def __init__(self):
        self.unary_map = VariableMap()
        self.pairwise_map = VariableMap()

    def num_vars(self):
        return self.pairwise_map.curr_index

    def __getitem__(self,key):
        if type(key[0]) is int: 
            return self.unary_map[key]      #point_to_index
        else:
            return self.pairwise_map[key]   #point_to_index


def apply_ball_and_intersect(radius,application_mask,intersection_mask,aux_mat=None):
    (h,w) = application_mask.shape    

    if aux_mat is None:
        aux_mat = MT.build_coordinates_matrix( h,w )

    for (y,x) in aux_mat[application_mask]:
        disk_mask = MT.generate_disk_region_mask(x,y,radius,shape=(h,w))
        intersection = MT.intersect(intersection_mask,disk_mask)

        for p in aux_mat[intersection]:            
            p = (int(p[0]),int(p[1]))
            yield p

def apply_ball_and_intersect_pair(radius,application_mask,intersection_mask,aux_mat=None):
    (h,w) = application_mask.shape    

    if aux_mat is None:
        aux_mat = MT.build_coordinates_matrix( h,w )

    for (y,x) in aux_mat[application_mask]:
        disk_mask = MT.generate_disk_region_mask(x,y,radius,shape=(h,w))
        intersection = MT.intersect(intersection_mask,disk_mask)

        for p1 in aux_mat[intersection]:            
            p1 = (int(p1[0]),int(p1[1]))

            for p2 in aux_mat[intersection]:            
                p2 = (int(p2[0]),int(p2[1]))            

                yield (p1,p2)            


def create_model_map(domain,radius,application_mask,opt_mask,aux_mat=None):
    mm = ModelMap()

    w,h = domain_dimensions(domain)
    

    for p in apply_ball_and_intersect(radius,application_mask,opt_mask,aux_mat):
        if p in mm.unary_map.point_to_index:
            continue
        mm.unary_map.update(p)
    

    num_unary_vars = mm.unary_map.curr_index
    mm.pairwise_map = VariableMap(mm.unary_map.curr_index)


    for (p1,p2) in apply_ball_and_intersect_pair(radius,application_mask,opt_mask,aux_mat):
        if p1==p2:
            continue
        if ( p1,p2 ) in mm.pairwise_map.point_to_index:
            continue

        mm.pairwise_map.update( (p1,p2),inc_index=False)
        mm.pairwise_map.update( (p2,p1),inc_index=True)


    return mm


def cvxopt(mm,unary_coefficients,pairwise_coefficients):
    num_vars = unary_coefficients.size+pairwise_coefficients.size
    c = np.zeros( num_vars)

    c[0:unary_coefficients.size] = unary_coefficients
    c[unary_coefficients.size:] = pairwise_coefficients

    num_constraints = 3*pairwise_coefficients.size + 2*num_vars

    A = np.zeros( (num_constraints,num_vars) )
    b = np.zeros( num_constraints )
    base_index = unary_coefficients.size

    #Linearization Constraints
    for i in range(pairwise_coefficients.size):
        pair_index = base_index + i
        
        p1,p2 = mm.pairwise_map.index_to_point[pair_index]

        i1 = mm.unary_map.point_to_index[p1]
        i2 = mm.unary_map.point_to_index[p2]

        A[3*i][pair_index] = 1
        A[3*i][i1] = -1
        b[3*i] = 0

        A[3*i+1][pair_index] = 1
        A[3*i+1][i2] = -1        
        b[3*i+1] = 0

        A[3*i+2][pair_index] = -1
        A[3*i+2][i1] = 1        
        A[3*i+2][i2] = 1       
        b[3*i+2] = 1

    #Between Zero and One Constraints
    base_index = 3*pairwise_coefficients.size
    for i in range(num_vars):
        A[base_index+2*i][i] = -1
        b[base_index+2*i] = 0

        A[base_index+2*i+1][i] = 1
        b[base_index+2*i+1] = 1        

    from cvxopt import matrix, solvers

    A = A.transpose()

    A = matrix( A.tolist() )
    b = matrix( b.tolist() )
    c = matrix( c.tolist() )

    sol=solvers.lp(c,A,b)    

    return sol['x']


def plot_mask(plt,mask,color,base_img):
    base_img[mask] = color
    plt.imshow(base_img,cmap="gray")


class Model:
    def __init__(self,input_data):
        self.radius = input_data["radius"]
        self.Q = input_data["Q"]
        self.domain = input_data["domain"]
        
        self.application_mask = input_data["application_mask"]
        self.opt_mask = input_data["optimization_mask"]
        self.trust_frg_mask = input_data["trust_frg_mask"]

        w,h = domain_dimensions(self.domain)
        self.aux_mat = MT.build_coordinates_matrix( h,w )
        
        self.mm = create_model_map(self.domain,self.radius,self.application_mask,self.opt_mask,self.aux_mat)
        
        self.num_unary_vars = self.mm.unary_map.distinct_count()
        self.num_pairwise_vars = self.mm.pairwise_map.distinct_count()

        self.num_vars = self.num_unary_vars + self.num_pairwise_vars

        self.unary_coefficients = np.zeros( self.num_unary_vars )
        self.pairwise_coefficients = np.zeros( self.num_pairwise_vars )

    def __coefficients__(self):
        for papp in self.aux_mat[self.application_mask]:
            (y,x) = papp
            disk_mask = MT.generate_disk_region_mask(x,y,self.radius,shape=self.application_mask.shape)
            
            trust_intersection = MT.intersect(disk_mask,self.trust_frg_mask)
            opt_intersection = MT.intersect(disk_mask,self.opt_mask)

            F = len(trust_intersection[trust_intersection])       
            for p1 in self.aux_mat[opt_intersection]:
                p1 = (int(p1[0]),int(p1[1]))
                i1 = self.mm[p1]

                self.unary_coefficients[ i1 ]  += -(self.Q-F-0.5)
          

        for (p1,p2) in apply_ball_and_intersect_pair(self.radius,self.application_mask,self.opt_mask,self.aux_mat):
            i1 = self.mm.unary_map.point_to_index[p1]            
            i2 = self.mm.unary_map.point_to_index[p2]            

            if p1==p2:
                continue

            if i1 > i2:
                continue

            self.pairwise_coefficients[ self.mm[ (p1,p2) ] - self.num_unary_vars ]+=1 



def build_new_input_from_solution(model,solution):

    trust_frg_mask = model.trust_frg_mask
    trust_frg_mask[solution==255] = True

    print(len(solution==255))

    optimization_mask = MT.binary_dilation(trust_frg_mask,MT.square(3))    
    ext_application_mask = MT.binary_dilation(optimization_mask,MT.square(3)) 

    optimization_mask = MT.minus(ext_application_mask,trust_frg_mask)

    (a,b),ext_application_mask = MT.get_boundary_mask( ext_application_mask )    

    optimization_mask = MT.minus(optimization_mask,ext_application_mask)

    (a,b),in_application_mask = MT.get_boundary_mask( trust_frg_mask )             

    application_mask = np.zeros( ext_application_mask.shape,dtype='bool' )
    application_mask[ext_application_mask] = True
    application_mask[in_application_mask] = True


    plt.clf()
    plt.subplot(1,2,1)
    base_img = np.zeros( trust_frg_mask.shape )
    plot_mask(plt,trust_frg_mask,255,base_img)
    plot_mask(plt,application_mask,128,base_img)
    plot_mask(plt,optimization_mask,64,base_img)

    plt.imshow(base_img,cmap="gray")

    plt.subplot(1,2,2)
    base_img = np.zeros( trust_frg_mask.shape )
    plot_mask(plt,trust_frg_mask,255,base_img)
    plot_mask(plt,application_mask,128,base_img)
    
    plt.imshow(base_img,cmap="gray")    
    plt.show()

    
    return {"radius":model.radius,
            "Q":model.Q,
            "domain": model.domain,
            "input_image":solution,
            "trust_frg_mask":trust_frg_mask,
            "application_mask":application_mask,
            "optimization_mask":optimization_mask}  

def solve(model):
    sol_x = cvxopt( model.mm,
                    model.unary_coefficients,
                    model.pairwise_coefficients)

    (w,h) = domain_dimensions(model.domain)
    
    image_solution = np.zeros( (h,w),dtype="uint8" )
    for i in range(model.num_unary_vars):        
        y,x = model.mm.unary_map.index_to_point[i]
        if sol_x[i] >= 0.5:                                    
            image_solution[y][x] = 255            
        elif sol_x[i] <= (0.001):            
            image_solution[y][x] = 0
        else:
            image_solution[y][x] = 64  #Unlabeled


    inverted_solution = np.zeros( (h,w) )
    for i in range(model.num_unary_vars):
        y,x = model.mm.unary_map.index_to_point[i]
        if image_solution[y][x]==255:
            inverted_solution[y][x] = 0
        elif image_solution[y][x]==64:
            inverted_solution[y][x] = 255
        else:
            inverted_solution[y][x] = 255

    final_solution = inverted_solution

    plot_mask(plt,model.trust_frg_mask,128,final_solution)
    plt.show()        

    solution = np.zeros( final_solution.shape,dtype='uint8')
    solution[final_solution>64] = 255

    return build_new_input_from_solution(model,solution)

def main():
    new_input = input()
    for i in range(10):
        print( len( new_input["trust_frg_mask"][ new_input["trust_frg_mask"] ] ) )
        model = Model( new_input )            
        model.__coefficients__()
        print(""" ***Model Map completed***\nUnary Terms: %d\nPairwise Terms: %d\nTotal: %d\n\n""" % (   model.num_unary_vars,
                                                                                                         model.num_pairwise_vars,
                                                                                                         model.num_vars) )
        new_input = solve(model)





if __name__=='__main__':
    main()
