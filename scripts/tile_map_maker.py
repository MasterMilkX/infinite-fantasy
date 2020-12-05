import numpy as np
from PIL import Image
import math
import os

class TileMapMaker():
	def __init__(self, map_path,tilesize=16):
		self.og_map = np.array(Image.open(map_path).convert('L'))		# read in the map image path and parse as integer array [0-255]
		self.tsize = tilesize

	#divides the map based on tiles
	def splitMap2Tiles(self, offX=0, offY=0):
		#assuming map is in 2d form

		#offset the original map
		newmap = self.og_map[offX:,offY:]

		#get w x h dimensions 
		width = int((newmap.shape[0])/self.tsize)
		height = int((newmap.shape[1])/self.tsize)

		tilemap = []
		#reshape the map based on the tilesize
		for w in range(width):
			for h in range(height):
				tile = newmap[w*self.tsize:(w+1)*self.tsize, h*(self.tsize):(h+1)*self.tsize]
				tilemap.append(tile)

		return np.array(tilemap).reshape(width,height,self.tsize,self.tsize)

	#convert color number based 2d array tile to string format
	def tile2Str(self, t):
		t2 = t.flatten()
		t2 = ",".join([str(hex(x)) for x in t2])		#make hex valued string for easy storage
		return t2

	#convert hex string tile to 2d color based tile
	def tile2Color(self, t_str):
		t = []
		i = 0
		tt = t_str.split(",")
		for x in range(self.tsize):
			tx = []
			for y in range(self.tsize):
				tx.append(int(tt[i],16))		#convert from hex based to decimal int based
				i+=1
			t.append(tx)
		return np.array(t,dtype='uint8')		#original format of tile

	#gets the occurrences of each tile
	def getTileOccurrences(self, tilemap):
		ts = tilemap.reshape(tilemap.shape[0]*tilemap.shape[1],16,16)
		occ = {}
		for t in ts:
			t2 = self.tile2Str(t)
			if t2 not in occ:		#add new tile
				occ[t2] = 0
			occ[t2] += 1			#increment found tile count
		return occ

	#determine how much of the tile hash would be dropped given the percentage
	def tileDropPercentage(self, tileHash, cutoff=5):
		tot = 0
		dropped = 0
		for t in tileHash.keys():
			v = tileHash[t]
			tot += v
			if v < cutoff:			#if not enough of this tile, drop it 
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

		#fill out the map replacing tiles with their indexed value from the
		for c in tilemap:
			cc = []
			for t in c:
				t2 = self.tile2Str(t)
				if not t2 in tileset:
					cc.append('x')			#special tile
				else:	
					cc.append(str(tileset[t2]))	#normal tile
			ascii_map.append(cc)

		return np.array(ascii_map)


	#divides the ascii map in a window size (tuple)
	def asciiWindows(self, ascii_map, ws):
		nw = (int(ascii_map.shape[1]/ws[0]),int(ascii_map.shape[0]/ws[1]))	#calculate number of windows to make
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

	#export the tile set to a png
	def exportTileSheet(self, tileset,name='tileset'):
		if not os.path.exists('tilesheets'):
			os.makedirs('tilesheets')

		tiles = sorted(tileset, key=lambda x: x[1])
		w = math.ceil(math.sqrt(len(tiles)))
		h = int(len(tiles)/w)

		#convert the tiles back to grayscale colors
		img = []
		i = 0
		for i in range(len(tiles)):
			img.append(self.tile2Color(tiles[i]))
		img = np.array(img)

		#make tilesheet
		img2 = []
		i=0
		for hi in range(h):
			r = []
			for wi in range(w):
				#place tiles horizontally next to each other
				if(len(r) == 0):
					r = img[i][:]
				else:
					r = np.hstack((r,img[i]))
				i+=1

			#stack tileset rows vertically 
			if(len(img2) == 0):
				img2 = r[:]
			else:
				img2 = np.vstack((img2,r))

		#export the tilesheet
		img_out = Image.fromarray(img2,'L')
		path = ("tilesheets/" + name + ".png")
		img_out.save(path)

		print("** Exported to '%s' @ (%d x %d) tiles ** " % (path, w, h))
		return

	#exports the ascii map generated from the tileset to a csv file
	def exportAsciiMap(self,ascii_map,name='ascii_map'):
		if not os.path.exists('ascii_maps'):
			os.makedirs('ascii_maps')

		path = "ascii_maps/" + name + ".csv"
		np.savetxt(path, np.asarray(ascii_map), delimiter=",",fmt='%s')
		print("** Exported to '%s' @ (%d x %d) map size ** " % (path, ascii_map.shape[0], ascii_map.shape[1]))
		return


#run demo for link's awakening map
if __name__ == "__main__":
	
	window_size = (10,8)

	TMM = TileMapMaker('maps/links_awakening.png')
	print("Original Map:\t%s" % str(TMM.og_map.shape))

	tm = TMM.splitMap2Tiles(offX=0,offY=0)
	print("Tile Set shape:\t%s" % str(tm.shape))

	oc = TMM.getTileOccurrences(tm)
	drop_perc = 5
	print("Drop % @ " + str(drop_perc) + " tiles: " + str(TMM.tileDropPercentage(oc,drop_perc)) + "%")

	tset = TMM.makeTileSet(oc)
	print("# tiles: " + str(len(tset)))

	am = TMM.makeAsciiMap(tset,tm)
	print("Ascii Map Shape: " + str(am.shape))
	#print(am[10:20,10:20])

	wm = TMM.asciiWindows(am, window_size)
	print("# windows: " + str(wm.shape[0]))
	#print(wm[0])

	TMM.exportTileSheet(tset,"links_awakening_tileset")
	TMM.exportAsciiMap(am,"links_awakening_ascii")
