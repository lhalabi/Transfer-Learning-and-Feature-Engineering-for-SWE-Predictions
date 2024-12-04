
# Modeling Spatial Distribution of Snow Water Equivalent using Transfer Learning across Mountainous Basins

Accurately estimating snow water equivalent (SWE) is crucial for understanding the impacts of climate change, urbanization, and population growth on water resources. High operational costs of lidar observations limit the frequency and coverage of SWE estimates at high spatial resolutions, leading to significant data gaps. We address this challenge with a transfer learning framework that leverages abundant SWE data from California to enhance predictions in Colorado, where data are scarce. From 2016 to 2019, the disparity in SWE data collection between these states was stark: 94 snowpack maps were recorded in California's Sierra Nevada versus only 12 in Colorado's Rocky Mountains. We hypothesized that geographic predictors (e.g., elevation and snowfall) would exhibit similar effects on SWE across these landscapes. By conducting an explanatory factor analysis, we validated this hypothesis and refined our transfer learning model, which incorporated data from California to predict SWE in Colorado. When compared with using data from Colorado alone, transfer learning improved the mean $R^2$ value from 0.43 to 0.56, indicating a significant enhancement of over 30\% in predictive accuracy. Such advancements underscore the potential of our framework to mitigate lidar data limitations, offering a valuable tool for water resource management amidst changing environmental conditions. 

**To reproduce this work please use python 3.6.1, Keras 2.6.0, and Tensorflow 2.6.2 versions.**
## Datasets

• Lidar data: we obtained lidar-derived SWE from the Airborne Snow Observatory (ASO) with a 50 m resolution (T. Painter, 2018). ASO data in Colorado consist of 12 maps across five basins from 2016 to 2019: Blue River (BR), Crested Butte (CB), Maroon/Castle Creek (CM), Gunnison-East River (GE), and Gunnison-Taylor River (GT) basins. ASO data in California are more frequent and consist of 94 maps across 13 basins, serving as a rich source for transfer learning. Following quality control measures, we considered 80 maps across 11 basins from 2013 to 2019: Cherry Eleanor (CE), Kings Canyon (KC), Kings North (KN), Lakes Basin (LB), LEE Vining Creek (LV), Merced River (MB), Rush Creek (RC), San Joaquin South Fork (SF), San Joaquin Main Fork (SJ), Tuolumne River (TB), Tuolumne + Cherry/Eleanor (TE).

• Gridded meteorological datasets: we obtained gridded estimates of daily preciptation and temperature from the Parameter elevation Relationships on Independent Slopes Model (PRISM) data (Daly et al., 2008) at a spatial resolution of 800 m. Using these datasets, we derived four meteorological variables that are strongly correlated to the spatial distribution of SWE: accumulated snow, sum positive degree days (PDD), accumulated precipitation, and mean seasonal temperature.

• Elevation maps: we obtained elevation maps from National Elevation Dataset (Gesch et al., 2018). These maps were used to extract topographic variables that influence snow melt and snow accumulation processes: slope, aspect, and elevation. 

These datasets were rescaled to 800 m and reprojected to a consistent coordinate system for consistency.

<div align=center><image src="./Figures/spatial_extent.jpg"></div>
<p align=center>
Figure 1: Spatial extent and frequency of lidar-derived SWE maps used in this study. Basin names are abbreviated for brevity. 
</p> 


| File | Description |
| ------------- | ------------- |
| `California CSV files/ Colorado CSV files` | Generated CSV datasets for Colorado and California. |
| `Final_processed_California_topography_SWE_snapshots.zip` | Processed ASO SWE maps and Topography files in California. |
| `Final_processed_Colorado_topography_SWE_snapshots.zip` | Processed ASO SWE maps and Topography files in Colorado. |

