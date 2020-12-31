import numpy as np
from PIL import Image
import math
import os
from tile_map_maker import TileMapMaker
from utils import *
import matplotlib.pyplot as plt
import csv
from sklearn.cluster import KMeans
import io

class TileClusterer():
	def __init__(self,ts, wm, map_path):
		self.map_name = os.path.basename(map_path).split(".")[0]
		self.tileset = ts
		self.windows = wm
		self.dirs = ['n','s','e','w']
		self.d_map = {'n':(-1,0),'s':(1,0),'w':(0,-1),'e':(0,1)}



	#combine array of data into one form (must be same number of instances)
	def combineData(self,dataArr):
		d = dataArr[0][:]
		for i in range(1,len(dataArr)):
			d = np.hstack((d,dataArr[i]))
		return d


	#####  ADJACENT TILE FEATURE  #####


	#return coordinates of a specific tile in a window map
	def tileCoords(self,t,m):
		s = np.where(m == t)
		l = list(zip(s[0],s[1]))
		return l

	#calculate the percentage of a tile that are the same as it in a given adjacent direction
	def adjSameTilePerc(self, t,d,windows):
		m = windows.reshape(np.prod(windows.shape[:2]),windows.shape[2],windows.shape[3])
		total_tiles = 0
		perc_amt = 0
		
		#iterate through every window
		for w in m:
			coords = self.tileCoords(t, w)
			
			#no tiles here
			n = len(coords)
			if n == 0:
				continue
			
			#get directional position of tile
			coords_alt = list(map(lambda x: tuple(np.add(x,self.d_map[d])), coords))  
			
			#count up same tiles in the adjacent direction
			p = 0
			for c in coords_alt:
				if c in coords:
					p += 1
			
			total_tiles += n
			perc_amt += p
			
		if total_tiles > 0: 
			return round(perc_amt/total_tiles,7)
		return 0

	#get all tile percentages
	def allAdjTilePerc(self,tset,w):
		atd = {}
		
		#get all tiles adjacent percentages
		for t in tset:
			td = {}
			for d in self.dirs:
				td[d] = self.adjSameTilePerc(t,d,w)
			atd[t] = td
		return atd



	#####   WINDOW FEATURE   #####

	#see if tile is in a window (get counts)
	def inWinBin(self,t,w):
		return 1 if t in w else 0
	#see if tile is in a window (get counts)
	def inWinMult(self,t,w):
		return (w==t).sum()

	#get all tile window locations
	def allTileWinLoc(self,tset,win):
		m = win.reshape(np.prod(win.shape[:2]),win.shape[2],win.shape[3])
		tloc = {}
		for t in tset:
			twin = []
			for w in m:
				twin.append(self.inWinBin(t,w))
			tloc[t] = twin
		return tloc



	#####   PARTIAL MIRROR TILE FEATURE   #####

	#check if tile matches certain %
	def partTileMatch(self,a,b,p):
		#check if same tile shape
		if len(a) != len(b):
			return 0
		
		m = 0         #count matches
		m1 = 0         #count mismatches
		
		p1 = 1.0-p
		
		t = (len(a)*p)    #number of tiles for matching
		t1 = (len(a)*p1)  #number of tiles for mismatching
		
		#iterate over all tiles
		for i in range(len(a)):
			if b[i] == a[i]:   #match
				m += 1
			else:                    #no match
				m1 += 1
					
			if m >= t:
				return 1    #enough matches
			if m1 > t1:
				return 0    #too many mismatches
				
		return 0


	#return whether mirror image of tile exists in the data (2 if mirror itself, 1 if mirror another tile)
	def almostMirrorTile(self,t,tset,p):
		a = tile2Color(t,16)   #convert to color for flipping
		
		flipH = tile2Str(np.flip(a,0))
		flipV = tile2Str(np.flip(a,1))
		flipD = tile2Str(np.flip(np.flip(a,0),1))
		
		#duplicates are auto mirror
		if flipH == t or flipV == t or flipD == t:
			return 2
		
		for i in tset:
			if self.partTileMatch(flipH,i,p):
				return 2 if i == t else 1
			if self.partTileMatch(flipV,i,p):
				return 2 if i == t else 1
			if self.partTileMatch(flipD,i,p):
				return 2 if i == t else 1
		return 0
		

	#find whether all tiles have mirror images
	def allTileAlmostMirror(self, tset,p):
		m = {}
		tile_ind = list(tset.keys())
		raw_tiles = list(tset.values())
		for t in tile_ind:
			m[str(t)] = self.almostMirrorTile(tset[t],raw_tiles,p)
		return m






	#forms k cluster groups from the tileset
	# (tile+window first - then mirror in largest cluster)
	def makeCascClusters(self,k,ts,wm):
		tiles = list(map(lambda x: str(x), ts.keys()))          #tile indexes

		#feature datas
		adj_tile_perc = self.allAdjTilePerc(tiles, wm)          #adjacent tiles
		tile_windows = self.allTileWinLoc(tiles,wm)             #window locations
		atam = self.allTileAlmostMirror(ts,0.7)                 #mirror data




		#create feature data arrays
		#convert dictionary directional percentages to list in consistent format
		exp1_data = []
		for t in tiles:
			l = []
			for i in self.dirs:
				l.append(adj_tile_perc[t][i])
			exp1_data.append(l)
		exp1_data = np.array(exp1_data)

		#convert dictionary window values to list in consistent format
		exp2_data = []
		for t in tiles:
			exp2_data.append(tile_windows[t])
		exp2_data = np.array(exp2_data)

		#convert dictionary partial mirror tiles to list in consistent format
		exp3_data = []
		for t in tiles:
			exp3_data.append([atam[t]])
		exp3_data = np.array(exp3_data)




		#make cluster first (tile + window)
		cluster = KMeans(n_clusters=k).fit(self.combineData([exp1_data,exp2_data]))
		l = list(cluster.labels_)

		big_label = max(set(l), key = l.count)

		#get all elements of biggest cluster
		b_cluster = {}
		ind = np.squeeze(np.where(l == big_label))

		#make subset clusters (partial tile mirror set) from largest dataset
		for t in ind:
			n = atam[str(t)]
			if n == 0:
				l[t] = l[t]
			else:
				l[t] = k+n-1




		tile_labels = dict(zip(tiles,l))
		return tile_labels


	#Convert a Matplotlib figure to a PIL Image and return it (from kotchwane)
	def fig2img(self,fig):
		buf = io.BytesIO()
		fig.savefig(buf)
		buf.seek(0)
		img = Image.open(buf)
		return img


	#shows the members of the cluster in image form
	def saveImgCluster(self,c,tiles):
		if not os.path.exists('clusters'):
			os.makedirs('clusters')


		#sort the images to their clusters
		clustSet = {}
		for k,v in c.items():
			if not v in clustSet:
				clustSet[v] = []
			clustSet[v].append(tiles[int(k)])


		#setup graph
		clustIMG = []

		#make mini graphs for each cluster tile
		for c, t in clustSet.items():
			#combine all cluster tiles together
			n = len(t)
			s = int(math.sqrt(n))
			w = math.ceil(n/s)
			fig = plt.figure(figsize=(5.0,5.0))
			for i in range(n):
				plt.subplot(s,w,i+1)
				plt.xticks([], figure=fig)
				plt.yticks([], figure=fig)
				plt.imshow(tile2Color(t[i],16).squeeze(),cmap='gray', figure=fig)

			fig.suptitle('Cluster ' + str(c))
			clustIMG.append(fig)


		#convert cluster set figures to image
		imglist = []
		for i in clustIMG:
			imglist.append(self.fig2img(i))


		#combine cluster images
		r = len(imglist)
		s = int(math.sqrt(r))
		w = math.ceil(r/s)
		plt.figure(figsize=(10.0*w,10.0*s))
		for i in range(r):
			plt.subplot(s,w,i+1)
			plt.xticks([])
			plt.yticks([])
			plt.imshow(imglist[i])

		path = ("clusters/" + self.map_name + "_cluster.png")
		plt.savefig(path)




	#export tiles indexing and their labels
	def saveTxtCluster(self,c):
		if not os.path.exists('clusters'):
			os.makedirs('clusters')
		w = csv.writer(open("clusters/" + self.map_name + "_cluster_labels.csv", "w"))
		for key, val in c.items():
			w.writerow([key, val])
		


if __name__ == "__main__":
	#get tileset and windows from tile map maker
	TMM = TileMapMaker('maps/links_awakening.png')
	window_size = (10,9)
	ts = TMM.importTileSet()
	wm = TMM.importWindows()


	TC = TileClusterer(ts,wm,'maps/links_awakening.png')
	c = TC.makeCascClusters(7,ts,wm)
	TC.saveImgCluster(c,ts)
	TC.saveTxtCluster(c)



