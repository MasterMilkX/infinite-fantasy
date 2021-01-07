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
	# uses features in first list as initial cluster then second feature set as next cluster
	# first cluster = k1, internal cluster size = k2
	# f1 = same adjacent tile, f2 = window location, f3 = partial mirror, f4 = pixel data
	def makeCascClusters(self,ts,wm,k=[10,3],feats=[[CL_F['PIX_REP']],[CL_F['WIN_LOC']]],weights=[1,1,1,1]):
		#error check
		if (len(feats[0]) == 0):
			print("## ERROR! Cannot have empty feature selection for first cluster! ##")
			return None
		if (k[0] == 0):
			print("## ERROR! Cannot have zero clusters for first cluster! ##")
			return None


		tiles = list(map(lambda x: str(x), ts.keys()))          #tile indexes

		#feature datas
		adj_tile_perc = self.allAdjTilePerc(tiles, wm)          #adjacent tiles
		tile_windows = self.allTileWinLoc(tiles,wm)             #window locations
		atam = self.allTileAlmostMirror(ts,0.7)                 #mirror data
		tile_feat = np.array(list(map(lambda x: tile2Color(x,16).flatten()/256,list(ts.values()))))		#raw tile data


		#create feature data arrays
		#convert dictionary directional percentages to list in consistent format
		exp1_data = []
		for t in tiles:
			l = []
			for i in self.dirs:
				l.append(adj_tile_perc[t][i])
			exp1_data.append(l)
		exp1_data = np.array(exp1_data)*weights[0]

		#convert dictionary window values to list in consistent format
		exp2_data = []
		for t in tiles:
			exp2_data.append(tile_windows[t])
		exp2_data = np.array(exp2_data)*weights[1]

		#convert dictionary partial mirror tiles to list in consistent format
		exp3_data = []
		for t in tiles:
			exp3_data.append([atam[t]])
		exp3_data = np.array(exp3_data)*weights[2]

		#get raw tile representations
		exp4_data = []
		for t in tile_feat:
			exp4_data.append(t)
		exp4_data = np.array(exp4_data)*weights[3]


		all_data = [exp1_data,exp2_data,exp3_data,exp4_data]
		first_data = []
		for i in feats[0]:
			first_data.append(all_data[i])

		#make cluster first (feature[0] selection)
		cluster = KMeans(n_clusters=k[0]).fit(self.combineData(first_data))
		l = list(cluster.labels_)

		#cascade features for biggest dataset
		if k[1] > 0 and len(feats[1]) != 0:
			big_label = max(set(l), key = l.count)

			#get all elements of biggest cluster
			b_cluster = {}
			ind = np.squeeze(np.where(l == big_label))


			#get dataset for big cluster items
			second_data_all = []
			for i in feats[1]:
				second_data_all.append(all_data[i])
			casc_feat = self.combineData(second_data_all)
			sec_data = casc_feat[ind]

			cluster2 = KMeans(n_clusters=k[1]).fit(sec_data)
			l2 = list(cluster2.labels_)

			#adjust labels from second dataset
			for i,a in zip(ind,l2):
				if a == 0:
					l[i] = l[i]
				else:
					l[i] = k[0]+a-1
		

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
	def exportImgCluster(self,c,tiles):
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

		print("** Exported clustered tiles PNG to '%s' with %d clusters ** " % (path, r))




	#export tiles indexing and their labels
	def exportTxtCluster(self,c):
		if not os.path.exists('clusters'):
			os.makedirs('clusters')
		csv_path = "clusters/" + self.map_name + "_cluster_labels.csv"
		w = csv.writer(open(csv_path, "w"))
		for key, val in c.items():
			w.writerow([key, val])

		print("** Exported cluster label CSV to '%s' with %d labels ** " % (csv_path, len(c)))
		


if __name__ == "__main__":
	demo = 1

	if demo == 1:
		#get tileset and windows from tile map maker
		TMM = TileMapMaker('maps/zelda_1.png')
		window_size = (16,11)
		ts = TMM.importTileSet()
		wm = TMM.importWindows()


		TC = TileClusterer(ts,wm,'maps/zelda_1.png')
		c = TC.makeCascClusters(ts,wm,k=[6,3],weights=[1,2,1,1])
		TC.exportImgCluster(c,ts)
		TC.exportTxtCluster(c)
	else:
		#get tileset and windows from tile map maker
		TMM = TileMapMaker('maps/links_awakening.png')
		window_size = (10,9)
		ts = TMM.importTileSet()
		wm = TMM.importWindows()


		TC = TileClusterer(ts,wm,'maps/links_awakening.png')
		c = TC.makeCascClusters(ts,wm,k=[10,3],feats=[[CL_F['PIX_REP'],CL_F['ADJ_TILE']],[CL_F['WIN_LOC']]])
		TC.exportImgCluster(c,ts)
		TC.exportTxtCluster(c)



