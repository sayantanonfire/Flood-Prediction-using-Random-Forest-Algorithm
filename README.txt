Geospatial AI: A Vector based GIS Framework for Robust Machine Learning Predictions

This repository presents a powerful and versatile Python-based framework for developing and deploying machine learning models for geospatial predictions. While demonstrated through the specific example of Flood Susceptibility Mapping, the core methodology is designed to be highly adaptable and applicable to a wide array of geospatial classification and regression problems.

Leveraging the geopandas library, this framework excels at seamlessly integrating geospatial vector data directly into machine learning workflows. This eliminates tedious data conversion steps, minimizes data distortions often associated with rasterization during preprocessing, and directly leverages the precise spatial locations of your input features for training. The final output is a ready-to-use geospatial raster prediction map, generated without the need for mandatory interpolation of input features.

1. Core Novelty & Generalizability: Vector-Native Geospatial ML
The traditional approach to geospatial machine learning often involves converting vector data (points, lines, polygons) into raster formats before feeding them to ML models. This project introduces a more efficient and robust paradigm:

Seamless Vector-to-ML Integration: The backbone of this framework is its ability to directly ingest and process geospatial vector point files (.shp) using geopandas. This means your raw spatial observations and their associated attributes (geographical, environmental, etc.) are fed to the ML model in their native vector format.

Reduced Time & Conversion Efforts: No more time-consuming conversions from vector to raster just for ML input. The code efficiently handles geographical coordinates and attributes within GeoDataFrames, significantly streamlining the data preparation phase.

Minimized Data Distortions: By operating directly on point features, the framework avoids the data distortion and loss of precision that can occur when interpolating or aggregating vector data into a raster grid before model training. The ML model learns from the most accurate representation of your spatial features.

Intelligent Prediction & Direct Raster Map Generation:

Point-Based Predictions: Machine learning predictions are made directly on the original set of input points, retaining their precise spatial coordinates.

Post-Prediction Rasterization (Flexible & On-Demand): Instead of pre-rasterizing features, this framework performs rasterization after the model has made its predictions. This allows for:

No Mandatory Input Interpolation: The ML model itself operates on the original, un-interpolated point values.

Enhanced Control over Output Maps: You gain complete control over the resolution (pixel_size) and projection of your final prediction maps, generating them in a GIS-ready raster format (e.g., GeoTIFF) at any desired resolution, without the need for further manual interpolation steps. This makes the output immediately usable in GIS software for visualization and spatial analysis.

2. Key Features
Direct Geospatial Data Ingestion: Reads ESRI Shapefiles (.shp) and their associated components directly using geopandas.

Automated Feature Engineering: Handles column renaming and intelligent dropping of irrelevant columns (Lat_Y, Long_X, Id, any_null, has_missin).

Robust Feature Selection & Multicollinearity Analysis: Includes Variance Inflation Factor (VIF) and Correlation Matrix analysis to identify and manage multicollinearity among features, ensuring stable and interpretable ML models.

Advanced Machine Learning: Utilizes powerful ensemble algorithms like Random Forest, optimized with GridSearchCV for effective hyperparameter tuning.

Comprehensive Model Evaluation: Provides standard metrics such as Classification Report, Confusion Matrix, ROC Curve, and Feature Importance to rigorously assess model performance.

Vector-to-Raster Prediction Mapping: Seamlessly converts point-based prediction probabilities into high-resolution, georeferenced raster maps (GeoTIFF), directly exportable for GIS applications.

Cross-Domain Applicability: Designed to be adaptable to various spatial prediction tasks where point-based features are available.

3. Methodology Overview
The workflow follows a logical progression, ensuring data quality and model robustness:

Data Loading & Initial Inspection: Geospatial vector point data (.shp) is loaded into geopandas GeoDataFrames. Basic data previews and column integrity checks are performed to ensure consistency between input and processed datasets.

Data Preprocessing & Feature Engineering: Irrelevant columns (e.g., Long_X, Lat_Y, Id, any_null, has_missin) are systematically dropped, and column names like Distance_D are standardized to Distance from Drainage for clarity.

Feature Selection & Multicollinearity Analysis: The numerical features are subjected to multicollinearity assessment using Variance Inflation Factor (VIF) and a Correlation Matrix. This crucial step helps identify highly correlated features, allowing for informed decisions on feature retention or removal to prevent model instability and improve interpretability. Visualizations (heatmaps, bar plots) aid in this analysis.

Machine Learning Model Training & Optimization:

The processed, spatially balanced data (Small_data_2.shp) is split into training and testing sets.

A machine learning pipeline, including StandardScaler for feature scaling and RandomForestClassifier, is configured.

GridSearchCV is employed to systematically search for the optimal hyperparameters for the Random Forest model, maximizing its performance based on a chosen scoring metric (e.g., f1-score).

Model Evaluation: The best-trained model's performance is rigorously evaluated on the test set. Key metrics generated include:

Classification Report: Provides precision, recall, F1-score, and support for each class.

Confusion Matrix: Visualizes the true positives, true negatives, false positives, and false negatives.

ROC Curve and AUC Score: Illustrates the model's ability to discriminate between classes across different thresholds.

Feature Importance: Quantifies the contribution of each input feature to the model's predictions, offering insights into the driving factors.

Large-Scale Prediction & Geospatial Export: The optimized model is then applied to the complete, original dataset (Large_data_Cleaned_2.shp) to generate prediction probabilities for every point. These probabilities are added as a new attribute (Flood_Prob in the example) to the GeoDataFrame, which is then exported as a new shapefile.

Raster Map Generation & Export:

The point-based probability predictions are transformed into a continuous raster surface using rasterio.features.rasterize.

The output raster's resolution (pixel_size) can be precisely defined (e.g., 40m, 45m).

The raster is re-projected to a widely compatible geographic CRS (e.g., WGS84 - EPSG:4326) for seamless integration with GIS software and web mapping applications.

The final result is a high-quality GeoTIFF (Flood_Prediction_40m_LatLong.tif) representing the susceptibility map.

4. Real-World Applications
This versatile framework can be adapted for numerous spatial prediction challenges across various fields:

Disaster Management:

Landslide Susceptibility Mapping: Predicting areas prone to landslides based on terrain, geology, and rainfall.

Wildfire Risk Assessment: Identifying zones at high risk of wildfires considering vegetation, climate, and human factors.

Drought Risk Mapping: Predicting drought-prone areas based on meteorological and hydrological indicators.

Environmental Science:

Pollution Hotspot Identification: Mapping areas with high concentrations of air, water, or soil pollutants.

Species Distribution Modeling: Predicting habitat suitability or presence/absence for specific flora and fauna.

Groundwater Potential Mapping: Identifying regions with high potential for groundwater resources, using geological, hydrological, and climatic factors.

Urban Planning & Infrastructure Development:

Crime Hotspot Prediction: Forecasting areas with high crime rates for targeted interventions and resource allocation.

Traffic Congestion Modeling: Predicting patterns of traffic congestion based on road network attributes, time of day, and urban density.

Optimal Site Selection: Identifying ideal locations for new infrastructure, facilities, or developments based on various spatial criteria.

Agriculture & Resource Management:

Crop Yield Prediction: Forecasting agricultural productivity based on soil quality, weather, and agricultural practices.

Mineral Prospectivity Mapping: Predicting potential locations for mineral deposits based on geological and geophysical surveys.

5. Code Structure and How to Use
The project is designed to be modular, with different stages of the workflow potentially residing in separate Jupyter notebooks or Python scripts for clarity and reusability.

5.1. Prerequisites
Python 3.x

Required Libraries:

geopandas

scikit-learn

rasterio

matplotlib

seaborn

numpy

pandas

statsmodels

shap

contextily (for optional basemaps in visualizations)

You can install these libraries using pip. It's highly recommended to use a virtual environment:

python -m venv venv
source venv/bin/activate # On Windows: .\venv\Scripts\activate
pip install geopandas scikit-learn rasterio matplotlib seaborn numpy pandas statsmodels shap contextily

5.2. Input Data Requirements
The framework requires geospatial vector point data as input.

File Format: ESRI Shapefiles (.shp) are the primary input format. Ensure that all associated files (.shx, .dbf, .prj, .cpg, etc.) are present alongside the .shp file.

Required Input Files:

Large_data_Cleaned_2.shp: This is your comprehensive dataset containing all feature variables and the target variable for the entire study area you wish to predict over.

Small_data_2.shp: This file is expected to be a spatially balanced and preprocessed subset of Large_data_Cleaned_2.shp. It is typically generated by a preceding spatial sampling script (like the one discussed in the previous context, Sampling_Data_Symetry.ipynb). This Small_data_2.shp is used specifically for training and validating the ML model.

Required Columns within Shapefiles:

Target_1 (integer): This is your dependent variable or target label. It should contain binary values (0 for the negative/non-event class, 1 for the positive/event class). For regression tasks, this would be your continuous target variable.

Feature Columns: All other columns in the attribute table will be treated as independent feature variables for your ML model (e.g., Aspect, Distance_D, Elevation, Plan_Curva, Flow_Accu, LULC, Slope, TWI, rainfall).

Coordinate Reference System (CRS): Ensure your input shapefiles have a defined CRS. The code handles re-projection to a projected CRS (e.g., EPSG:32643 for UTM) for accurate distance calculations and then re-projects the final raster output to EPSG:4326 (WGS84 Lat/Long) for broader compatibility.

5.3. Setup and Installation
Clone the repository:

git clone https://github.com/sayantanonfire/Flood-Prediction-using-Random-Forest-Algorithm.git
cd [Flood-Prediction-using-Random-Forest-Algorithm.git]

(Replace [Your-Repo-Name] with the actual name of your repository)

Place your input data: Ensure Large_data_Cleaned_2.shp and Small_data_2.shp (along with their companion files: .shx, .dbf, .prj, etc.) are placed in the root directory of your cloned repository.

Install dependencies: Follow the steps in Section 5.1 to create a virtual environment and install all necessary Python libraries.

5.4. Execution Steps
The typical workflow involves executing the relevant Python scripts or Jupyter notebooks in a sequential manner.

Data Loading and Preprocessing:

This initial part of the code (often found at the beginning of the main ML script, e.g., Random_Forest_2.py or Random_Forest_2.ipynb) loads Large_data_Cleaned_2.shp and Small_data_2.shp.

It performs initial data inspection, renames columns (Distance_D to Distance from Drainage), and drops unnecessary columns (Lat_Y, Long_X, Id, any_null, has_missin).

Action: Run this section first.

Feature Selection & Multicollinearity Analysis:

This section (usually following data loading) calculates VIF scores and generates a correlation matrix for the features in Small_data_2.shp.

It produces visualizations (heatmap and VIF bar plots) to aid in understanding feature relationships and identifying multicollinearity.

Action: Review the outputs to ensure features are suitable for modeling.

ML Model Training & Evaluation:

The Small_data_2.shp (without geometry and target) is split into X_train, X_test, y_train, y_test.

A Pipeline is set up with StandardScaler and RandomForestClassifier.

GridSearchCV performs hyperparameter tuning on the training data.

Model performance is evaluated using classification_report, confusion_matrix, roc_curve, and feature_importances_.

Action: Run this section to train and evaluate your ML model.

Prediction & Raster Map Generation:

The best-trained model makes predictions (predict_proba) on the full Large_data_Cleaned_2.shp dataset (excluding geometry and target).

Prediction probabilities (Flood_Prob) are added back to gdf_large.

The gdf_large is re-projected to a suitable projected CRS (e.g., EPSG:32643).

rasterio.features.rasterize converts the point probabilities into a raster.

The raster is saved as a temporary GeoTIFF, then re-projected to WGS84 (EPSG:4326) and saved as the final output GeoTIFF.

Finally, the raster map is visualized using matplotlib and rasterio.plot.show.

Action: Run this final section to generate your prediction map.

5.5. Key Parameters to Tune
You can adjust the following parameters within your Python scripts to customize the workflow for your specific dataset and requirements:

Machine Learning Model Hyperparameters:

For RandomForestClassifier (within param_grid for GridSearchCV):

rf__n_estimators: Number of decision trees in the forest (e.g., [100, 200]).

rf__max_depth: Maximum depth of each tree (e.g., [5, 10, None]). None means nodes are expanded until all leaves are pure or contain less than min_samples_split samples.

rf__min_samples_split: Minimum number of samples required to split an internal node (e.g., [2, 5]).

rf__min_samples_leaf: Minimum number of samples required to be at a leaf node (e.g., [1, 2]).

Rasterization Settings:

pixel_size: The desired resolution of the output raster map in meters (e.g., 40, 45). Choose this based on your data density and the desired spatial resolution of the final map.

epsg code for projected CRS: In the rasterization steps (gdf_large.to_crs(epsg=...)), ensure you use the correct UTM or local projected CRS for your study area. For instance, 32643 is UTM Zone 43N. This should match the CRS of your input data for accurate transformation.

Visualization: Adjust cmap (colormap), title, figsize, vmin, vmax in matplotlib and seaborn plotting functions to fine-tune the visual representation of your results.

5.6. Output Files
The code will generate the following key output files in your repository directory:

Predicted_Flood_Susceptibility.shp: A new ESRI Shapefile containing all original points from Large_data_Cleaned_2.shp, with an additional attribute (Flood_Prob in the example) representing the predicted probability of susceptibility for each point.

Flood_Prediction_40m_LatLong.tif (or Flood_Prediction_45m.tif depending on pixel_size): A GeoTIFF raster file, which is a continuous grid representing the predicted probability surface across the study area. This file is georeferenced and re-projected to WGS84 (Lat/Long), making it universally compatible with GIS software (e.g., QGIS, ArcGIS) and web mapping platforms.

Temporary .tif files (e.g., Flood_Prediction_40m_TEMP.tif) may be created during the raster reprojection process; these can be safely ignored or deleted after the final .tif is generated.

6. Project Files
The core logic of this framework is distributed across several Jupyter notebooks (or can be combined into Python scripts):

Sampling_Data_Symetry.ipynb: This notebook (or script) is designed to handle the initial data balancing and preparation, generating the Small_data_2.shp file from the larger Large_data_Cleaned_2.shp using techniques like spatial buffering and K-Means clustering.

Random_Forest_2.ipynb / Random_Forest_Workspace.ipynb: These notebooks likely contain the primary machine learning workflow, including feature analysis (VIF, correlation), model training, hyperparameter tuning, evaluation, and the final prediction and rasterization steps.

1 Static_Data.ipynb / 2 Dynamic_Data.ipynb: These might be notebooks for initial data exploration, cleaning, or feature engineering for different types of variables (e.g., static terrain features vs. dynamic rainfall data).

Deep_Learning Model.ipynb / Ground_water_MLPNN.ipynb: These indicate the framework's flexibility to incorporate other machine learning models, including deep learning (Multi-Layer Perceptron Neural Networks specifically for groundwater).

Back_Calculator.ipynb: Potentially a script for inverse modeling, sensitivity analysis, or scenario testing.

7. Contributing
Feel free to fork this repository, open issues for bug reports or feature requests, or submit pull requests with improvements. Contributions are highly welcome!

8. License
[Choose a license (e.g., MIT License) and add details here]

9. Contact
For any queries, collaborations, or further discussions, feel free to reach out:

Creator: Sayantan Mandal
Email: sayantanonfire@gmail.com