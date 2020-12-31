import numpy as np


#convert hex string tile to 2d color based tile
def tile2Color(t_str,tsize):
    t = []
    i = 0
    tt = t_str.split(",")
    for x in range(tsize):
        tx = []
        for y in range(tsize):
            tx.append(int(tt[i],16))		#convert from hex based to decimal int based
            i+=1
        t.append(tx)
    return np.array(t,dtype='uint8')		#original format of tile

#convert color number based 2d array tile to string format
def tile2Str(t):
    t2 = t.flatten()
    t2 = ",".join([str(hex(x)) for x in t2])		#make hex valued string for easy storage
    return t2