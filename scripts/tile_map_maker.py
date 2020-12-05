import numpy as np
from PIL import Image
import math
import os
from tqdm import tqdm
import json

class TileMapMaker():
	def __init__(self, map_path,tilesize=16):
		self.map_name = os.path.basename(map_path).split(".")[0]
		self.og_map = np.array(Image.open(map_path).convert('L'))		# read in the map image path and parse as integer array [0-255]
		self.tsize = tilesize

	#returns the og map without the border
	def removeBorder(self, ws, thick=1):
		bmap = self.og_map[:]

		#make border indexes
		hbor = []
		for b in range(thick):
			hbor += list(range(b,bmap.shape[0],(ws[1]*self.tsize)+thick))
		vbor = []
		for b in range(thick):
			vbor += list(range(b,bmap.shape[1],(ws[0]*self.tsize)+thick))

		#remove the border from the map and return
		bmap = np.delete(bmap, hbor, axis=0)
		bmap = np.delete(bmap, vbor, axis=1)

		return bmap

	#divides the map based on tiles
	def splitMap2Tiles(self, offX=0, offY=0, border=0, ws=None):
		spMap = self.og_map[:]

		#assuming map is in 2d form
		if border > 0 and ws != None:
			spMap = self.removeBorder(ws,border)

		#offset the original map
		newmap = spMap[offX:,offY:]

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
		 
		b = np.array(b).reshape(nw[1],nw[0],ws[1],ws[0])
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
	def exportAsciiMap(self,ascii_map,name='ascii_map',extension='csv',delim=','):
		if not os.path.exists('ascii_maps'):
			os.makedirs('ascii_maps')

		path = "ascii_maps/" + name + "." + extension
		np.savetxt(path, np.asarray(ascii_map), delimiter=delim,fmt='%s')
		print("** Exported to '%s' @ (%d x %d) map size ** " % (path, ascii_map.shape[0], ascii_map.shape[1]))
		return

	#export the window data to json format
	def exportWindows(self, windows,name='windows'):
		#check if folder exists first
		if not os.path.exists('map_windows'):
			os.makedirs('map_windows')

		i = 0
		wd = {}
		wm = []
		#assign each window an index
		for y in range(windows.shape[0]):
			r = []
			for x in range(windows.shape[1]):
				wd[i] = windows[y][x].tolist()
				i+=1
				r.append(i)
			wm.append(r)

		#export to path
		s = {'windows':wd,'map':wm}
		path = "map_windows/" + name + ".json"
		with open(path, "w") as outfile:  
			json.dump(s, outfile) 

		print("** Exported to '%s' @ (%d x %d) windows ** " % (path, windows.shape[1], windows.shape[0]))

		return

	#finds the best tile set and occurence set based on calculated offset
	def findBestTileSplit(self,drop_tiles,border=0,ws=None):
		bestOff = (0,0)
		oc = None
		tm = None
		lowDrop = 100

		#go through every pixel combination
		with tqdm(total=(self.tsize**2)) as pbar:
			for a in range(self.tsize):
				for b in range(self.tsize):
					t = self.splitMap2Tiles(offX=a,offY=b,border=border,ws=ws)		#get tiles split from the original map
					o = self.getTileOccurrences(t)				#get tile occurrences
					dp = self.tileDropPercentage(o,drop_tiles)

					pbar.update(1)	#update progress bar

					#if lowest drop percentage seen, save the offset and tiles
					if dp < lowDrop:
						bestOff = (a,b)
						lowDrop = dp
						oc = o
						tm = t

					#if drop percentage under 5% then stop looking altogether
					if lowDrop < 5.0:
						return tm, oc, bestOff, lowDrop

		#return best found
		return tm, oc, bestOff, lowDrop

	#makes ascii map, windows, and tilesheet based calculated offset 
	def run(self,tilesize,ws,drop_tiles=5,border=0,calcOffSet=False,DEBUG=False):
		if DEBUG:
			print("-- Map:\t\t" + str(self.map_name))

		self.tsize = tilesize
		if DEBUG:
			print("-- Tile size:\t" + str(self.tsize) + " x " + str(self.tsize))
			print("-- Border:\t" + str(border))

		
		if calcOffSet:
			if DEBUG:
				print(" > Calculating Offset **")

			#get the best split of tiles
			tm, oc, off, drop = self.findBestTileSplit(drop_tiles,border,ws) 
			if DEBUG:
				print("-- Drop %:\t" + str(drop) +"%")
				print("-- Map Offset:\t" + str(off))  
		
		else:
			if DEBUG:
				print(" > USING OFFSET (0,0) **")

			#get the tileset and tile occurrences (assume offset = (0,0))
			tm = self.splitMap2Tiles(border=border,ws=ws)
			oc = self.getTileOccurrences(tm)

		#make the tileset with associated indexes 
		tset = self.makeTileSet(oc)
		if DEBUG:
			print("-- # tiles:\t" + str(len(tset)))

		#create the ascii map using the original map and the newly made tileset
		am = self.makeAsciiMap(tset, tm)
		if DEBUG:
			print("-- Ascii Map:\t" + str(am.shape))

		#create the windows sets
		wm = self.asciiWindows(am,ws)
		if DEBUG:
			print("-- # windows:\t" + str(wm.shape[0]*wm.shape[1]))
			print("")

		#export the tileset and ascii map
		self.exportTileSheet(tset,self.map_name+"_tileset")
		self.exportAsciiMap(am,self.map_name+"_ascii")
		self.exportWindows(wm,self.map_name+"_windows")

		
#run demo for link's awakening map
if __name__ == "__main__":
	'''
	window_size = (16,10)
	border = 1
	TMM = TileMapMaker('maps/zelda_1.png')
	'''

	window_size = (10,9)
	border = 0
	TMM = TileMapMaker('maps/links_awakening.png')

	TMM.run(16,window_size,border=border,DEBUG=True,calcOffSet=False)
	

