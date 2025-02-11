import numpy as np
from sklearn.cluster import KMeans

def get_grid_idcs(points, num_points, seed=None):

    points = np.array(points)
    
    # Apply k-means clustering to points with (num_points) clusters
    kmeans = KMeans(n_clusters=num_points, random_state=seed).fit(points)
    
    # Get the cluster centers
    all_centers = kmeans.cluster_centers_
    
    # Find the indices of the points closest to the centers
    selected_indices = []
    for center in all_centers:
        closest_index = np.argmin(np.linalg.norm(points - center, axis=1))
        if closest_index not in selected_indices:  # Avoid duplicates
            selected_indices.append(closest_index)


    return selected_indices


def gen_reg_grid(search_radius, spatial_resolution, angular_resolution):
    # Generate a regular grid of points within a search radius, spaced by spatial_resolution and angular_resolution
    n = np.ceil(search_radius/spatial_resolution)
    x_vals = np.arange(-n*spatial_resolution, n*spatial_resolution + spatial_resolution, spatial_resolution)
    y_vals = np.arange(-n*spatial_resolution, n*spatial_resolution + spatial_resolution, spatial_resolution)
    angles = np.arange(0, 181, angular_resolution)

    X, Y, Z = np.meshgrid(angles, x_vals, y_vals)
    
    grid = np.stack((X.flatten(),Y.flatten(), Z.flatten())).T
    keep = np.sqrt(grid[:,1]**2+grid[:,2]**2) <= search_radius
    grid = grid[keep]

    return grid