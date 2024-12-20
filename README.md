
# Modeling Spatial Distribution of Snow Water Equivalent using Transfer Learning across Mountainous Basins
Accurately estimating snow water equivalent (SWE) is crucial for understanding the impacts of climate change, urbanization, and population growth on water resources. High operational costs of lidar observations limit the frequency and coverage of SWE estimates at high spatial resolutions, leading to significant data gaps. We address this challenge with a transfer learning framework that leverages abundant SWE data from California to enhance predictions in Colorado, where data are scarce. From 2016 to 2019, the disparity in SWE data collection between these states was stark: 94 snowpack maps were recorded in California's Sierra Nevada versus only 12 in Colorado's Rocky Mountains. We hypothesized that geographic predictors (e.g., elevation and snowfall) would exhibit similar effects on SWE across these landscapes. By conducting an explanatory factor analysis, we validated this hypothesis and refined our transfer learning model, which incorporated data based on 80 snowpack maps from California to predict SWE in Colorado. Compared to using data from Colorado alone, transfer learning improved the mean R$^2$ value from 0.45 to 0.54, representing a significant enhancement of 20\% in predictive accuracy, while reducing bias from 0.24 to 0.17 (a 30\% decrease). Such advancements underscore the potential of our framework to mitigate lidar data limitations, offering a valuable tool for water resource management amidst changing environmental conditions. 

**To reproduce this work please use python 3.6.1, Keras 2.6.0, and Tensorflow 2.6.2 versions.**
## Datasets

• Lidar data: we obtained lidar-derived SWE from the Airborne Snow Observatory (ASO) with a 50 m resolution (T. Painter, 2018). ASO data in Colorado consist of 12 maps across five basins from 2016 to 2019: Blue River (BR), Crested Butte (CB), Maroon/Castle Creek (CM), Gunnison-East River (GE), and Gunnison-Taylor River (GT) basins. ASO data in California are more frequent and consist of 94 maps across 13 basins, serving as a rich source for transfer learning. Following quality control measures, we considered 80 maps across 11 basins from 2013 to 2019: Cherry Eleanor (CE), Kings Canyon (KC), Kings North (KN), Lakes Basin (LB), LEE Vining Creek (LV), Merced River (MB), Rush Creek (RC), San Joaquin South Fork (SF), San Joaquin Main Fork (SJ), Tuolumne River (TB), Tuolumne + Cherry/Eleanor (TE).

• Gridded meteorological datasets: we obtained gridded estimates of daily preciptation and temperature from the Parameter elevation Relationships on Independent Slopes Model (PRISM) data (Daly et al., 2008) at a spatial resolution of 800 m. Using these datasets, we derived four meteorological variables that are strongly correlated to the spatial distribution of SWE: accumulated snow, sum positive degree days (PDD), accumulated precipitation, and mean seasonal temperature.

• Elevation maps: we obtained elevation maps from National Elevation Dataset at a resolutionof 10 m (Gesch et al., 2018). These maps were used to extract topographic variables that influence snow melt and snow accumulation processes: slope, aspect, and elevation. 

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

## Transfer Learning and Benchmark Models
We adopted a feed-forward Artificial Neural Network (ANN) architecture, initially training a base model on the source data which corresponds to California's 80 SWE maps. Subsequently, we considered four different modeling approaches to adapt the base model to perform the target task of predicting SWE in Colorado (Figure 2). The first two approaches were: 

• Model TL1: This involved freezing the shallower layers (preventing their weights from changing) and retraining only the deeper layers.

• Model TL2: This involved freezing all the weights of the model, removing deeper layers, and adding new layers whose weights were trained on the target data.

The two approaches were picked because the deeper layers help capture the higher order complexities in the relationship between input features and the output, while the shallower layers generally capture coarser and simpler relationships. Complex relationships in the deeper layers of the base model may be more particular to the source task, while the information in the shallower layers may be more easily generalized to the target task. Additionally, it is important to mention that in our analysis, permutation feature importance applied to the transfer learning models revealed that elevation, accumulated snow, and accumulated precipitation were not predominant factors during training. This finding contrasted with our EFA results, which highlighted elevation, snow, and precipitation as critical variables in determining SWE values in Colorado. Therefore, we developed a third transfer learning approach where the importance of these variables was prescribed:

• Model TL1_W: Uses the architecture and hyper-parameters of the best performing TL1 models, however the elevation, snow, and precipitation features where multiplied by weights greater than 1. The weights were treated as hyperparameters and the optimal weights were found using Optuna, an open-source automatic hyperparameter optimization framework.

• Model TL2_W: Uses the architecture and hyper-parameters of the best performing TL2 models, however the elevation, snow, and precipitation features where multiplied by weights greater than 1. The weights were treated as hyperparameters and the optimal weights were found using Optuna.

• Local Models: the performance of transfer learning models was benchmarked against local models trained only on data from Colorado. This helps to validate the added value of transfer learning in improving SWE prediction accuracy. We considered two versions of local models: Local 1 considers scaled input variables per the usual machine learning practice, while Local 1_W prescribes importance to elevation, snow, and precipitation in a manner similar to model TL1_W and TL2_W.

All models, except the base models, were trained between September and November 2024 using the specified Python and Keras versions. The architectures for the local models were optimized by Ax in a prior study (2022) and saved for consistency. To ensure version alignment for accurate comparison, the saved architectures were cloned, and new local models were retrained. Two sets of Local 1 models are available on GitHub: one serves as the architecture template for the retrained models, while the second set comprises the models being compared in this study.

| File | Description |
| ------------- | ------------- |
| `Base_models.zip` | The 5 trained Base models. The Base model used in transfer learning is Base model 3. |
| `Local_1.tar` | 12 Local 1 models|
| `Local_1_W.tar` | 12 Local 1_W models|
| `TL1.tar` | 12 TL1 models |
| `TL1_W.tar.zip` | 12 TL1_W models |
| `TL2.tar` | 12 TL2 models |
| `TL2_W.tar` | 12 TL2_W models |
| `Colorado_ScaledLMs.zip` | 12 older Local 1 models that serve as the architecture template of Local 1 and Local 1_W models |


## Factor Analysis Results

We developed four explanatory factor analysis (EFA) models: two EFA models describing Colorado datasets (one each for March-April and June), and two EFA models describing California datasets (one each for March-April and June). EFA models captured between 0.64 to 0.72 of the total variance in our dataset. All models were able to capture a large proportion of the variance in elevation, accumulated snow, accumulated precipitation, sum PDD (with the exception for March-April in Colorado), and $T_{mean}$. For SWE, the four EFA models were able to capture 0.75, 0.78, 0.74, and 0.57 of the variance.

To further investigate the regional and seasonal variability in how the predictor variables affect SWE, we resorted to factor loading plots shown in Figures 3 and 4. These plots illustrate the association between variables and latent factors. Each latent factor can be considered to be a proxy or latent representation of underlying physical phenomena that influence snowpack. Factors are arranged based on the amount of variance they capture from the data, listed in descending order. Three factors were found to fit Colorado datasets best, while two factors yielded the best fit for California datasets.

Colorado in March/April         |  California in March/April
:-------------------------:|:-------------------------:
![](/Figures/COL_FA_loadingplot_winter_equal_y.jpg) |  ![](/Figures/CA_FA_loadingplot_winter_equal_y.jpg )

<p align=center>
Figure 3: EFA Loading plots illustrating the principal factors for Colorado and California datasets in March/April. The x-axis represents the variable name, while the y-axis represents the variable loading.
</p> 

Colorado in June         |  California in June
:-------------------------:|:-------------------------:
![](/Figures/COL_FA_loadingplot_summer_equal_y.jpg) |  ![](/Figures/CA_FA_loadingplot_summer_equal_y.jpg )

<p align=center>
Figure 4: EFA Loading plots illustrating the principal factors for Colorado and California datasets in June. The x-axis represents the variable name, while the y-axis represents the variable loading.
</p> 

The analysis reveals consistent patterns across both Colorado and California, with elevation and accumulated snow consistently driving high SWE values throughout different months. Additionally, low temperatures are consistently related to high SWE values in both regions across various time periods. This indicates that transfer learning could be implemented to predict SWE in Colorado using data from California. However, disparities emerge in the influence of precipitation and temperature-driven processes, with Colorado showing a stronger dependence on precipitation and California exhibiting a more pronounced sensitivity to temperature-related factors, particularly in March-April. There is also a disparity in the effect of slope on SWE values in Colorado and California. These point to differences in higher-order complexities in relationships between the predictors and SWE across Colorado and California. The importance of elevation, accumulated snow, and accumulated precipitation in Colorado, coupled with the reduced significance of precipitation in California, motivated the application of weights to these three variables in our modeling approach. Lastly, there is notable seasonal variability in SWE in Colorado, as evidenced by the more pronounced importance of temperature-related processes in June compared to March.

## SWE Prediction Results

Table 2 and Table 3 summarize the $R^2$ and $KGE$ values for local models and TL models. The analysis of model performance across KGE, $R^2$, and bias metrics demonstrates that TL1W and TL2W consistently
<div align=center><image src="./Figures/SWE_results.png"></div>
<p align=center>
Table 1: $R^2$ values for modeling SWE. 
</p> 

