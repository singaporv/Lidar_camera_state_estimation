# -*- coding: utf-8 -*-
import numpy as np
import open3d as op
import argparse

def main():
	parser = argparse.ArgumentParser(description='DBSCAN segmentation algorithm')
	parser.add_argument("--file", dest="filename", required=True, help='Provide a .pcd file')
	parser.add_argument("--eps", dest="eps", required=True, type=float, help='radius threshold', default=1)
	parser.add_argument("--minPts", dest="minPoints", required=True, type=int, help='clustering threshold points', default=50)
	parser.add_argument("--kitti", dest="useKitti", required=True)
	parser.add_argument("--sklearn", dest="useSklearn", default="True")
	args = parser.parse_args()

	# Loading Data using open3D
	pcd = op.io.read_point_cloud(args.filename) # Load data here

	if(args.useSklearn == str(True)):
		try:
			import sklearn.neighbors
			import dbscan_2 as dbscan
			downsampleRatio = np.array(pcd.points).shape[0] / 1e4
		except Exception as e:
			print(e)
			print("Please 'pip3 install scikit-learn, else set --sklearn False")
			exit()
	else:
		import dbscan
		downsampleRatio = np.array(pcd.points).shape[0] / 1e3

	# Sample down the data if large input is given
	pcd = pcd.uniform_down_sample(every_k_points=int(downsampleRatio))

	# Converting pcd into python array
	pcdArray = np.asarray(pcd.points)

	ground_list = []    
	above_ground = []  

	if(args.useKitti == "True"):
		# Ground plane segregation
		for point in pcdArray:
		  if point[2] > -1.5:     # considering points above -1.5 m  of LiDAR as above the ground
		    above_ground.append(point)
		  else:
		    ground_list.append(point)
		ground = np.asarray(ground_list)
		above_ground = np.asarray(above_ground)

	# Clustering algorithm using DBSCAN. Getting all the labels for segmentation   
	labels = dbscan.DBSCAN(above_ground, args.eps, args.minPoints)
	labels = np.array(labels)

	# Array to store the segments
	pcdArrayList = []

	# Color list (Randomly called to color clusters)
	colorList = [[0,1,0],[0,0,1],
	             [1,1,0],[1,0,1],[0,1,1],[1,0,0]]

	# Creating cluster cloud segments
	for label in range(len(set(labels))):
	    newPcdArray = above_ground[np.where(labels == label)[0]]
	    # Converting the 3D points into open3D based structure
	    pcdPoints = op.utility.Vector3dVector(newPcdArray)
	    newPcd = op.geometry.PointCloud()
	    newPcd.points = pcdPoints
	    color = colorList[np.random.randint(6)]
	    colorArr =  np.full((newPcdArray.shape[0],3), color)
	    newPcd.colors = op.utility.Vector3dVector(colorArr) 
	    pcdArrayList.append(newPcd)
	    
	    # The labels have noises (-1), which is not a cluster
	    if newPcdArray.size:
	      # Get the extremities of the point cloud segment data
	      x_max = max(newPcdArray[:,0])
	      x_min = min(newPcdArray[:,0])
	      y_max = max(newPcdArray[:,1])
	      y_min = min(newPcdArray[:,1])
	      z_max = max(newPcdArray[:,2])
	      z_min = min(newPcdArray[:,2])

	      # Create a bounding box around the point cloud
	      cube = [
	        [x_min, y_min, z_min],
	        [x_max, y_min, z_min],
	        [x_min, y_max, z_min],
	        [x_max, y_max, z_min],
	        [x_min, y_min, z_max],
	        [x_max, y_min, z_max],
	        [x_min, y_max, z_max],
	        [x_max, y_max, z_max]]

	      # The below creates relations to join cube points to form bounding boz
	      lines = [[0,1],[0,2],[1,3],[2,3],
	              [4,5],[4,6],[5,7],[6,7],
	              [0,4],[1,5],[2,6],[3,7]]            
	      
	      # Creeate line object
	      line_set = op.geometry.LineSet()

	      # Convert list to open3D objects
	      line_set.points = op.utility.Vector3dVector(cube)
	      line_set.lines = op.utility.Vector2iVector(lines)
	      colors = [[0, 0, 0] for i in range(len(lines))]
	      line_set.colors = op.utility.Vector3dVector(colors)

	      # Adding lines to the point cloud for visualization
	      pcdArrayList.extend([line_set])

	
	if(args.useKitti == "True"):
		# Now adding the ground plane back
		newPcd = op.geometry.PointCloud()
		newPcd.points = pcdPoints
		newPcd.points = op.utility.Vector3dVector(ground)
		newPcd.paint_uniform_color([0.5, 0.5, 0.5])

		# Add ground plan to the existing list
		pcdArrayList.extend([newPcd]) 

	# Display the results   
	op.visualization.draw_geometries(pcdArrayList)

if __name__ == "__main__":
	main()
