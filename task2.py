
# Imports

import xarray as xr
import plotly.graph_objects as go
import numpy as np
import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from element import members   # element_id -> [start_node, end_node]
from node import nodes        # node_id -> [x, y, z]


# Load dataset

DATASET_PATH = r"C:\Users\ADMIN\Downloads\screening_task.nc"

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError("screening_task.nc file not found")

ds = xr.open_dataset(DATASET_PATH)
F = ds["forces"]


# Girder definitions (as given in task)

girders = {
    "Girder 1": [13, 22, 31, 40, 49, 58, 67, 76, 81],
    "Girder 2": [14, 23, 32, 41, 50, 59, 68, 77, 82],
    "Girder 3": [15, 24, 33, 42, 51, 60, 69, 78, 83],
    "Girder 4": [16, 25, 34, 43, 52, 61, 70, 79, 84],
    "Girder 5": [17, 26, 35, 44, 53, 62, 71, 80, 85],
}


# Function to generate 3D diagram

def create_diagram(diagram_type="SFD"):
    """
    diagram_type:
        'SFD' -> Shear Force Diagram
        'BMD' -> Bending Moment Diagram
    """

    fig = go.Figure()

    # Select force components
    if diagram_type == "SFD":
        comp_i, comp_j = "Vy_i", "Vy_j"
        title = "3D Shear Force Diagram (SFD)"
        color_line = "blue"
        filename_png = "results/3D_SFD.png"
    else:
        comp_i, comp_j = "Mz_i", "Mz_j"
        title = "3D Bending Moment Diagram (BMD)"
        color_line = "red"
        filename_png = "results/3D_BMD.png"

   
    # Compute scale 
    
    all_vals = []
    for elems in girders.values():
        for e in elems:
            all_vals.append(abs(F.sel(Element=e, Component=comp_i).item()))
            all_vals.append(abs(F.sel(Element=e, Component=comp_j).item()))

    max_val = max(all_vals)
    scale = 5.0 / max_val 
    
# ================== K-MEANS ON FORCE VALUES (TASK 2) ==================
    force_values = []

    for elems in girders.values():
       for e in elems:
        force_values.append(abs(F.sel(Element=e, Component=comp_i).item()))
        force_values.append(abs(F.sel(Element=e, Component=comp_j).item()))

    force_array = np.array(force_values).reshape(-1, 1)

    
    kmeans3d = KMeans(n_clusters=3, random_state=0)
    kmeans3d.fit(force_array)

    cluster_labels = kmeans3d.labels_

    
      # target vertical height ≈ 5 units
    
    matrix = []

    for gname, elements in girders.items():
        row = []
        for e in elements:
          val = abs(F.sel(Element=e, Component=comp_i).item())
          row.append(val)
        matrix.append(row)


    plt.figure()
    sns.heatmap(matrix, cmap="coolwarm")
    plt.title(f"{diagram_type} Intensity Across Girders")
    plt.xlabel("Span Elements")
    plt.ylabel("Girders")
    plt.show()

    # Loop through girders
   
    for gname, elements in girders.items():

        base_x, base_y, base_z = [], [], []
        diag_x, diag_y, diag_z = [], [], []

        for k, elem in enumerate(elements):
            n1, n2 = members[elem]

            x1, y1, z1 = nodes[n1]
            x2, y2, z2 = nodes[n2]

            v1 = F.sel(Element=elem, Component=comp_i).item()
            v2 = F.sel(Element=elem, Component=comp_j).item()

            # Get cluster index for coloring
            cluster_index_1 = kmeans3d.predict([[abs(v1)]])[0]
            cluster_index_2 = kmeans3d.predict([[abs(v2)]])[0]

            colorscale = ['blue', 'orange', 'red']  # low, medium, high
            color1 = colorscale[cluster_index_1]
            color2 = colorscale[cluster_index_2] 


            # Base girder geometry
            if k == 0:
                base_x.extend([x1, x2])
                base_y.extend([y1, y2])
                base_z.extend([z1, z2])
                diag_x.extend([x1, x2])
                diag_z.extend([z1, z2])
            else:
                base_x.append(x2)
                base_y.append(y2)
                base_z.append(z2)
                diag_x.append(x2)
                diag_z.append(z2)

            
            #  CORRECT BMD DIRECTION FOR MIDAS STYLE
            if diagram_type == "BMD":
                # MIDAS-style: BMD always downward (negative Y)
                if k == 0:
                    diag_y.extend([
                        y1 - abs(v1) * scale,
                        y2 - abs(v2) * scale
                    ])
                else:
                    diag_y.append(
                        y2 - abs(v2) * scale
                    )
            else:
                # SFD follows sign convention
                if k == 0:
                    diag_y.extend([
                        y1 + v1 * scale,
                        y2 + v2 * scale
                    ])
                else:
                    diag_y.append(
                        y2 + v2 * scale
                    )
           

        # Base structure
        fig.add_trace(go.Scatter3d(
            x=base_x, y=base_y, z=base_z,
            mode="lines",
            line=dict(color="black", width=4),
            name=f"{gname} - Structure",
            showlegend=(gname == "Girder 1")
        ))

       # Draw segment with cluster color
        fig.add_trace(go.Scatter3d(
         x=[x1, x2],
         y=[diag_y[-2], diag_y[-1]],
         z=[z1, z2],
         mode="lines",
         line=dict(color=color1, width=4),
         showlegend=False

        ))


        # Vertical line for this segment
        fig.add_trace(go.Scatter3d(
         x=[x1, x1],
         y=[y1, diag_y[-2]],
         z=[z1, z1],
         mode="lines",
         line=dict(color=color1, width=1),
         showlegend=False
         ))

        fig.add_trace(go.Scatter3d(
         x=[x2, x2],
         y=[y2, diag_y[-1]],
         z=[z2, z2],
         mode="lines",
         line=dict(color=color1, width=1),
         showlegend=False
         ))


    
    # Layout (MIDAS-like)
    
    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
           xaxis=dict(
    title=dict(text="X (Longitudinal)",font=dict(size=16)
        
    ),
    tickfont=dict(size=13),
    showline=True,
    linewidth=2,
    linecolor="black",
    gridcolor="lightgray"
),

yaxis=dict(
    title=dict(
        text="Vertical extrusion",
        font=dict(size=16)
        
    ),
    tickfont=dict(size=13),
    showline=True,
    linewidth=2,
    linecolor="black",
    gridcolor="lightgray"
),

zaxis=dict(
    title=dict(
        text="Z (Transverse)",
        font=dict(size=16)
       
    ),
    tickfont=dict(size=13),
    showline=True,
    linewidth=2,
    linecolor="black",
    gridcolor="lightgray"
),
        ),
        

    margin=dict(l=120, r=120, b=80, t=80)
    
)
    os.makedirs("results", exist_ok=True)
    fig.write_image(filename_png, engine="kaleido", scale=2)
    print(f"Saved {filename_png}")


# Run both diagrams

if __name__ == "__main__":
    create_diagram("SFD")
    create_diagram("BMD")


