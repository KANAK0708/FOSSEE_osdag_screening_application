#**FOSSEE Screening Tasks – Structural Analysis & Visualization**
This project focuses on extracting structural force results from an Osdag NetCDF file and converting them into clear, meaningful Shear Force Diagrams (SFD) and Bending Moment Diagrams (BMD) using Python.

The work is divided into two parts:

Task‑1: 2D visualization and analysis of the central girder
Task‑2: Full 3D visualization of all girders in the bridge

The aim is to understand how forces behave in the structure and to validate Osdag results using programming, visualization, and basic data analytics.

##🚀 Features
✅ Task‑1 — 2D Analysis of Central Girder
Reads force data directly from the NetCDF file using Xarray
Converts element forces into continuous data along the girder
Plots 2D SFD and BMD using Matplotlib
Highlights maximum and minimum force locations
Performs basic statistical analysis (mean, standard deviation)
Uses K‑Means clustering to classify force regions (low/medium/high)
Shows data distribution using histogram and boxplot
Studies the relationship between shear and moment

✅ Task‑2 — 3D Analysis of Entire Bridge
Maps element forces to actual 3D node coordinates
Creates 3D SFD and BMD using Plotly
Visualizes how forces vary along the span and across girders
Uses K‑Means clustering to color high and low force regions
Generates a heatmap to show moment intensity across the bridge
Produces diagrams similar to professional structural software

📁 Files in the Repository
File	
task1_2D_SFD_BMD.py	#Code for Task‑1 (2D plots + statistics + clustering)
task2_3D_SFD_BMD.py	#Code for Task‑2 (3D plots + clustering + heatmap)
element.py	#Element to node connectivity information
node.py #	Node coordinate information (x, y, z)
screening_task.nc	#NetCDF file exported from Osdag
results/	#Folder where all generated images are saved

⚙️ How to Run
Step 1 — Install required libraries
pip install xarray matplotlib plotly numpy seaborn pandas scikit-learn kaleido

Step 2 — Keep these files in the same folder
screening_task.nc
element.py
node.py

Step 3 — Run Task‑1
python task1_2D_SFD_BMD.py

This will generate:

BMD_central.png
SFD_central.png

Step 4 — Run Task‑2
python task2_3D_SFD_BMD.py

This will generate:

3D_SFD.png
3D_BMD.png

Heatmap visualization

📘 Summary of Approach
Instead of manually inspecting numbers from Osdag output, this project uses Python to:
Extract force data
Map it to actual bridge geometry
Visualize it in 2D and 3D
Apply simple statistical and machine learning techniques to better understand force behavior
Task‑1 helps understand force variation along a single girder, while Task‑2 expands this understanding to the entire bridge in 3D space.
This shows how programming and data visualization can be effectively used in structural engineering analysis.

📜 License
This project is created for academic learning and demonstration purposes.
Feel free to use and modify it for educational and research work.
