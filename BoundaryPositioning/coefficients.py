#coding:utf-8

import numpy as np

import tools as MT

def application_domain(domain,radius):
    lb = domain[0]
    ub = domain[1]

    return ( (lb[0]+radius,lb[1]+radius), (ub[0]-radius,ub[1]-radius) )


def domain_iterator(domain):
    lb = domain[0]
    ub = domain[1]

    for x in range(lb[0],ub[0]):
        for y in range(lb[1],ub[1]):
            yield (x,y)

class VariableMap:
    def __init__(self):
        self.curr_index = 0

        self.pointToIndex = {}
        self.indexToPoint = {}

    def update(self,point):
        self.point_to_index[ point ] = self.curr_index
        self.index_to_point[ self.curr_index ] = point        

        self.curr_index+=1

class ModelMap:
    def __init__(self):
        self.xMap = VariableMap()
        self.yMap = VariableMap()

        self.zMap = VariableMap()
        self.wMap = VariableMap()



def create_variable_map(index_to_point,point_to_index,domain,app_domain,radius):
    index = 0

    mm = ModelMap()

    for pt in domain_iterator(domain):    
        mm.xMap.update( pt )
        mm.yMap.update( pt )

    single_vars = index

    aux_mat = MT.build_coordinates_matrix( domain[1][1],domain[1][0] )
    for center_pt in domain_iterator(app_domain):
        disk_mask = MT.generate_disk_region_mask(x,y,est_disk_radius,shape=domain[1])

        for pair_pt in aux_mat[disk_mask]:
            mm.zMap.update( (center_pt,pair_pt) )



def main():
    est_disk_radius = 3
    domain = ( (0,0), (8,8) )
    app_domain = application_domain(domain,est_disk_radius)


    index_to_point = {}
    point_to_index = {}
    create_variable_map(index_to_point,point_to_index,domain)

    num_vars = len(index_to_point.keys())
    coefficients = np.zeros( num_vars )

    print( num_vars )
    print( coefficients[ point_to_index[ (1,1) ] ] )

    aux_mat = MT.build_coordinates_matrix( domain[1][1],domain[1][0] )
    for (x,y) in domain_iterator(app_domain):
        disk_mask = MT.generate_disk_region_mask(x,y,est_disk_radius,shape=domain[1])

        for pt in aux_mat[disk_mask]:
            pt = ( int(pt[0]), int(pt[1]) )
            coefficients[ point_to_index[ pt ] ]+=1






if __name__=='__main__':
    main()