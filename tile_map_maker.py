import numpy as np
from PIL import Image

class TileMapMaker():
	def __init__(self, map_path,tilesize=16):
		self.og_map = np.array(Image.open(map_path).convert('L'))		# read in the map image path and parse as integer array [0-255]
		self.ts = tilesize

	#divides the map based on tiles
	def splitMap2Tiles(self, offX=0, offY=0):
		#assuming map is in 2d form

		#offset the original map
		newmap = self.og_map[offX:,offY:]

		#get w x h dimensions 
		width = int((newmap.shape[0])/self.ts)
		height = int((newmap.shape[1])/self.ts)

		tilemap = []
		#reshape the map based on the tilesize
		for w in range(width):
			for h in range(height):
				tile = newmap[w*self.ts:(w+1)*self.ts, h*(self.ts):(h+1)*self.ts]
				tilemap.append(tile)

		return np.array(tilemap).reshape(width,height,self.ts,self.ts)

	def tile2Str(self, t):
		t2 = t.flatten()
		t2 = ",".join([str(hex(x)) for x in t2])
		return t2

	#gets the occurrences of each tile
	def getTileOccurrences(self, tilemap):
		ts = tilemap.reshape(tilemap.shape[0]*tilemap.shape[1],16,16)
		occ = {}
		for t in ts:
			t2 = self.tile2Str(t)
			if t2 not in occ:
				occ[t2] = 0
			occ[t2] += 1
		return occ

	#determine how much of the tile hash would be dropped given the percentage
	def tileDropPercentage(self, tileHash, cutoff=5):
		tot = 0
		dropped = 0
		for t in tileHash.keys():
			v = tileHash[t]
			tot += v
			if v < cutoff:
				dropped += v
		return (dropped / tot)*100

	#make the tileset associated with number values (based on seeing tiles with 5 or more occurences)
	def makeTileSet(self, tilehash, cutoff=5):
		#remove tiles that do not make the cutoff
		ts = dict(filter(lambda x: x[1] >= cutoff,tilehash.items()))

		#sort from most to least occuring (at cutoff) tiles
		ts = dict(sorted(ts.items(), key=lambda x: x[1], reverse=True))

		#give indexes to the tiles (key = tile, value = index)
		t2 = {}
		tiles = list(ts.keys())
		for i in range(len(tiles)):
			t2[tiles[i]] = i

		return t2

	#makes an ascii map using the tileset generated
	def makeAsciiMap(self, tileset, tilemap):
		ascii_map = []
		map_dim = (tilemap.shape[0],tilemap[1])
		for c in tilemap:
			cc = []
			for t in c:
				t2 = self.tile2Str(t)
				if not t2 in tileset:
					cc.append('x')			#special tile
				else:	
					cc.append(tileset[t2])	#normal tile
			ascii_map.append(cc)

		return np.array(ascii_map)


	#divides the ascii map in a window size (tuple)
	def asciiWindows(self, ascii_map, ws):
		nw = (int(ascii_map.shape[1]/ws[0]),int(ascii_map.shape[0]/ws[1]))
		print(nw)
		a = ascii_map

		#make windows
		b = []
		for y in range(nw[1]):
		    for x in range(nw[0]):
		        window = []
		        ox = x*ws[0]
		        oy = y*ws[1]
		        for h in range(ws[1]):
		            for w in range(ws[0]):
		                tile = (a[(oy+h):(oy+(h+1)),(ox+w):(ox+(w+1))])
		                window.append(tile)
		        b.append(window)
		 
		b = np.array(b).reshape(np.prod(nw),ws[1],ws[0])
		return b


window_size = (10,8)

TMM = TileMapMaker('maps/links_awakening.png')
print("Original Map:\t%s" % str(TMM.og_map.shape))

tm = TMM.splitMap2Tiles(offX=0,offY=0)
print("Tile Set shape:\t%s" % str(tm.shape))

oc = TMM.getTileOccurrences(tm)
'''
print("Occurence set:")
for k, v in oc.items(): 
	print(k[:12] + " : " + str(v))
'''
drop_perc = 5
print("Drop % @ " + str(drop_perc) + " tiles: " + str(TMM.tileDropPercentage(oc,drop_perc)) + "%")

tset = TMM.makeTileSet(oc)
print("# tiles: " + str(len(tset)))
# for k, v in tset.items(): 
# 	print(k[:12] + " : " + str(v))

am = TMM.makeAsciiMap(tset,tm)
print(am.shape)

wm = TMM.asciiWindows(am, window_size)
print(wm[0])
