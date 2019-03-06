import numpy as np

def read_route(txt_path):
    read_raw_route=open(txt_path,'r')
    raw_route=read_raw_route.readlines()
    read_raw_route.close()
    route_co=np.zeros((len(raw_route),40+1,2))  #route_co[number of routes][steps][x,y]
    route=np.zeros((len(raw_route),40,1))
 
    for i in range(len(route_co)):
      each_raw_route=str(raw_route[i])
      each_raw_route=each_raw_route.split('/')
      for j in range(40+1):
        coordinate=each_raw_route[j].split(',')
        route_co[i][j][0]=coordinate[0]
        route_co[i][j][1]=coordinate[1]

    for i in range(route_co.shape[0]):
      for j in range(route_co.shape[1]):
        if j == 40:
          break
        if route_co[i][j+1][0] == route_co[i][j][0] and route_co[i][j+1][1] == route_co[i][j][1]:
          route[i][j][0] = 0
        if route_co[i][j+1][0] == route_co[i][j][0] and route_co[i][j+1][1] > route_co[i][j][1]:
          route[i][j][0] = 1
        if route_co[i][j+1][0] > route_co[i][j][0] and route_co[i][j+1][1] == route_co[i][j][1]:
          route[i][j][0] = 2
        if route_co[i][j+1][0] < route_co[i][j][0] and route_co[i][j+1][1] == route_co[i][j][1]:
          route[i][j][0] = 3

    route = np.reshape(route,(10,40))

    return route


def read_origin_des(txt_path):
    read_origin_des=open(txt_path,'r')
    raw_origin_des=read_origin_des.readlines()
    read_origin_des.close()
    origin_des=np.zeros((len(raw_origin_des),4))  #route[number of routes][current step][x,y,d_d,d_y]

    for i in range(len(raw_origin_des)):
      each_raw_origin_des=str(raw_origin_des[i])
      each_raw_origin_des=each_raw_origin_des.split('/')
      for k in range(40):
        for j in range(2):
          coordinate=each_raw_origin_des[j].split(',')
          origin_des[i][j][0]=coordinate[0]
          origin_des[i][j][1]=coordinate[1]



    return origin_des
    

"""
def convert_to_directioon(route):
    
    for i in range(route.shape[0]):
      for j in range(route.shape[1])
        if route[i][j][0] == route[i][j+1][0]  and route[i][j][1] == route[i][j+1][1]:
          direction_route[i][j][0] = 0
        else if route[i][j][0] == route[i][j+1][0]  and route[i][j][1] == route[i][j+1][1]:
        else if
"""










      
