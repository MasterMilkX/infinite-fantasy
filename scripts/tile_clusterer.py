import numpy as np
from PIL import Image
import math
import os
from tqdm import tqdm
from tile_map_maker import TileMapMaker

class TileClusterer():
	def __init__(ts, wm):
		self.tileset = ts
		self.windows = wm

	#forms k cluster groups from the tileset based on the windows
	def makeClusters(self,k,adj=4):
		return

	def showClusters(self,clust):
		return


if __name__ == "__main__()":
	#get tileset and windows from tile map maker
	TMM = TileMapMaker('maps/zelda_1.png')
	window_size = (10,9)
	ts = TMM.importTileSet()
	wm = TMM.importWindows()


	TC = TileClusterer(ts,wm)