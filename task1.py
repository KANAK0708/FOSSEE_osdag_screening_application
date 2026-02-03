#Task-1: 2D Shear Force Diagram (SFD) and Bending Moment Diagram (BMD)
#STEP 0: Imports
import xarray as xr
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.cluster import KMeans

from element import members
from node import nodes



# STEP 1: Load the Xarray dataset

# This opens the NetCDF file generated from Osdag analysis
ds = xr.open_dataset(r"C:\Users\ADMIN\Downloads\screening_task.nc")

#The forces are stored in a single data variable called 'forces' with dims like ('Element','Component')
F = ds["forces"]


# STEP 2: central girder elements---given directly in problem statement
central_elements = [15, 24, 33, 42, 51, 60, 69, 78, 83]

# STEP 3: Prepare lists to store results

x_pos = []   # Position along bridge length
Mz = []      # Bending Moment values
Vy = []      # Shear Force values


first=True

# STEP 4: Extract forces element-wise

for elem in central_elements:
   
    n_i,n_j=members[elem]

    x_i=nodes[n_i][0]
    x_j=nodes[n_j][0]                                           

    # Mz -> Bending Moment (Mz)
    mz_i = F.sel(Element=elem, Component="Mz_i").item()
    mz_j = F.sel(Element=elem, Component="Mz_j").item()

    # Vy -> Shear Force (Vy)
    vy_i = F.sel(Element=elem, Component="Vy_i").item()
    vy_j = F.sel(Element=elem, Component="Vy_j").item()

    if first:
        # For the first element, add only the starting node to avoid duplication
        x_pos.append(x_i)
        Mz.append(mz_i)
        Vy.append(vy_i)
        first=False

    # Append values in correct order
    x_pos.append(x_j)
    Mz.append(mz_j)
    Vy.append(vy_j)

# ================== K-MEANS CLUSTERING ON BENDING MOMENT ==================

# Convert to numpy array for ML
data = np.array(Mz).reshape(-1, 1)

# Create KMeans model (3 clusters: low, medium, high)
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(data)

clusters = kmeans.labels_
# Print cluster centers
print("Mz Mean:", np.mean(Mz))
print("Mz Std Dev:", np.std(Mz))
print("Vy Mean:", np.mean(Vy))
print("Vy Std Dev:", np.std(Vy))


# STEP 5: Create results folder

os.makedirs("results", exist_ok=True)

# Find max and min for Mz and Vy for annotation
Mz_max = max(Mz)
Mz_min = min(Mz)

x_Mz_max = x_pos[Mz.index(Mz_max)]
x_Mz_min = x_pos[Mz.index(Mz_min)]



# STEP 6: Plot Bending Moment Diagram (BMD)

plt.figure()
plt.scatter(x_pos, Mz, c=clusters, cmap='viridis', s=80)
plt.plot(x_pos, Mz, color="black", linewidth=1)

# Smooth curve fitting
z = np.polyfit(x_pos, Mz, 3)
p = np.poly1d(z)
plt.plot(x_pos, p(x_pos), '--')

plt.colorbar(label="Cluster Group (Low → High Moment)")

plt.axhline(0,color="black",linewidth=1)
# Mark max and min points
plt.scatter(x_Mz_max, Mz_max, color="green", zorder=5)
plt.scatter(x_Mz_min, Mz_min, color="purple", zorder=5)

plt.text(x_Mz_max, Mz_max,
         f"Max = {Mz_max:.2f} kN-m",
         ha='left', va='bottom')

plt.text(x_Mz_min, Mz_min,
         f"Min = {Mz_min:.2f} kN-m",
         ha='left', va='top')
plt.title("Bending Moment Diagram (BMD) - Central Longitudinal Girder")
plt.xlabel("Distance(m)")
plt.ylabel("Bending Moment Mz (kN-m)")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/BMD_central.png", dpi=220)

plt.show()


# ================== BOXPLOT OF BENDING MOMENT ==================
plt.figure()
plt.boxplot(Mz)
plt.title("Boxplot of Bending Moment")
plt.show()

# ================== HISTOGRAM OF BENDING MOMENT ==================
plt.figure()
plt.hist(Mz, bins=8, color='red')
plt.title("Distribution of Bending Moment")
plt.show()



# Find max and min for Vy for annotation
Vy_max = max(Vy)
Vy_min = min(Vy)

x_Vy_max = x_pos[Vy.index(Vy_max)]
x_Vy_min = x_pos[Vy.index(Vy_min)]


# ================== K-MEANS CLUSTERING ON SHEAR FORCE ==================
data_vy = np.array(Vy).reshape(-1, 1)

kmeans_vy = KMeans(n_clusters=3, random_state=0)
kmeans_vy.fit(data_vy)

clusters_vy = kmeans_vy.labels_



# STEP 7: Plot Shear Force Diagram (SFD)

plt.figure()
plt.scatter(x_pos, Vy, c=clusters_vy, cmap='plasma', s=80)
plt.plot(x_pos, Vy, color="black", linewidth=1)
plt.colorbar(label="Cluster Group (Low → High Shear)")
plt.axhline(0,color="black",linewidth=1)
# Mark max and min points
plt.scatter(x_Vy_max, Vy_max, color="green", zorder=5)
plt.scatter(x_Vy_min, Vy_min, color="purple", zorder=5)

plt.text(x_Vy_max, Vy_max,
         f"Max = {Vy_max:.2f} kN",
         ha='left', va='bottom')

plt.text(x_Vy_min, Vy_min,
         f"Min = {Vy_min:.2f} kN",
         ha='left', va='top')
plt.title("Shear Force Diagram (SFD) - Central Longitudinal Girder")
plt.xlabel("Distance(m)")
plt.ylabel("Shear Force (Vy)")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/SFD_central.png", dpi=220)
plt.show()


# ================== CORRELATION BETWEEN Vy AND Mz ==================
plt.figure()
plt.scatter(Vy, Mz)
plt.xlabel("Shear Vy")
plt.ylabel("Moment Mz")
plt.title("Vy vs Mz Relation")
plt.show()
