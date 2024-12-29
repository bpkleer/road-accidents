# Road accidents in France (2005-2023)

*Work in progress*

In this project, I explore road accidents in mainland France and the severity per accident. I focus on the location of the accidents to enhance road safety solutions. Thereby, results could support multiple areas, from insurances to urban planning, by providing insights into high risk zones regarding the severity of an accident. 

## Overall aim

Predicting severity score based (not only) on geographical information such as lattitude and longitude can help within various areas. With predictions one might identify areas with higher risks, indicating problems on that specific road/place. 

The identidifed high-risk areas could be used for policy purposes to inspect these areas and improve road safety along these. Furthermore this could also be used for insurance purposes, such as calculating the proximity of new customers to these high-risk regions and calculating the customer's insurance based on proximity to high-risk areas.

Machine Learning Model: The ML model doesn’t need the proximity calculation because it’s focused on predicting the severity of accidents based on various features (including spatial ones). The model uses lattitude/longitude directly as part of the input, along with other features, to learn patterns and predict severity.

The severity scores predicted by the model will be useful for identifying high-risk areas. Once you have those areas (based on severity), you can calculate proximity for new customers. 

## Data
The data is released by [data.gouv.fr](data.gouv.fr). For the years 2005-2021, there exists a pooled data set, and for the years 2022 and 2023 single files. The following two data sets are available and of interest for the aim of this project:

- characteristics (`caract`): It includes features on the road accidents' environment like weather, date of time, or geographical data. Overall it includes $1231695$ rows and $16$ columns. 
- places (`lieux`): It includes information on the area around the road accident. The data set has $1143471$ rows and $19$ columns/features. 
- `usagers`: It includes information on the persons involved in the accident (drivers, passengers, or pedestrians). From this data set, I just used the feature `gravity` to calculate a severity score per accident. 

Regarding road accidents, there exists one more data sets (`vehicule`) which has no information for the problem addressed above. The data set inherits only car-specific information (such as motor etc.) and, therefore, I don't use it. 

Since I deal with an overall score of the accident, I only take into consideration features that describe the accident (not car or individual information). Hence, I shrink the data set to one line per accident, keeping only the accident information (such as location). With this approach, I keep in the merged un-engineered data set $2,399,342$ rows. In addition, I only deal with mainland France and deleted cases where information on lattitude or longitude was not given or could not be retrieved in gradients which shrinkens the data set further to $126,732$ rows.

## Reasoning for models
The target of this project is numerical and most features are categorical with sometimes up to 9 groups in the original data set. The only two non-categorical features are lattitude and longitude (in gradients), although these could be transformed into numerical values, it is not expected that these two have a linear relationship to the target. Hence, I focussed on tree-based models since these are better at capturing non-linear relationships (especially reagarding lattitude and longitude) and the categorical features. 

I opted for the following three models to test:

1. Random Forest
2. XGBoost
3. CatBoost

## Feature Engineering

### Target

I created a target variable based on the persons involved in the accident and the individual severity. In the training data, the extent of persons involved in an accident ranges from 1 to 84. However, 75\% of the values are within 1 to 3, indicating outliers/extreme values (4.33\%) of big accidents in the data set. 

For constructing a score for each accident, I refer to the [Abbreviated Injury Scale (AIS)](https://en.wikipedia.org/wiki/Abbreviated_Injury_Scale) which classifies severity of injuries. However, the AIS is more detailed and therefore I can only relate the categories of this data set to the AIS:

| gravity             | weight | AIS Score                                                                      |
|---------------------|--------|--------------------------------------------------------------------------------|
| No information      | 0      | No equivalent                                                                  |
| Uninjured           | 0      | No registered injury                                                           |
| Light injury        | 1      | Minor injury (1)                                                               |
| Injured in hospital | 3.5    | Moderate injury (2) or Serours injury (3) or severe injury (4) or critical (5) <br> All four kinds are kinds for a hospital  |
| Killed              | 6      | Fatal injury (6)                                                               |

For building the target, it is important to regard that there should be extra weight to the severity if persons are killed. Otherwise an accident with 5 lightly injured perople would get the same score as an accident with 1 person killed. Therefore, we apply a penalty to the mean value, if a person is killed:

$$Severity Score = \frac{1}{n} \sum_{i=1}^n is_i  + weight_{killed} * pk^{0.4} + weight_{hospital} * ph^{0.2}$$

$$is$$: individual score
$$pk$$: persons killed
$$ph$$: persons in hospital

Therefore, an accident that involves 5 persons, of whom 2 were killed, 1 was injured in hospital, 1 had light injury and 1 was uninjured, the overall score is: $\frac{0*1 + 1*1 + 3.5*1 + 6*2}{5} + (2 * 6)^0.4 + (1*3.5)^0.2 \approx 7.29$

### Features


In the `cleaning` process of the data, I created first the target `severity_score` and deleted all helper features for this creation as well as `gravity` (the original indicator of individual gravity). 

Second, I dropped duplicates rows of each individual, since I'm only interested in the overall accident's severity score. 

The geographical features `long` (longitude) and `lat` (lattitude) showed problems in the cleaning process. Besides standard problems such as strings and unwanted signs in the strings (i.e, `-`), some entries did not adhere to the gradients of France. I used GeoPandas and tried several transforming from different sources, however, none of these transformations led to successful transformation. Unfortunately, there was no information given which EPSG was used. Therefore, I needed to delete these rows, since they are not on the same scale as the gradients and, hence, follow a different numerical scale. Since I could not find the coding of the non-gradient scale, I had to opt for the gradient scale, since I could not interpret a scaling that I'm not aware of. 

None of these steps in the cleaning violates the rule against data leakage, since the transformation is just on the variable on its own. 

| Feature | Cleaning | Engineering |
| ---- | ---------- | --------- |
| `lat` | inconsistent format (not all are in gradients)<br>tried several other formats, none was working<br>dropped lines with values out of France range (in gradients) | - |
| `long` | inconsistent format (not all are in gradients)<br>tried several other formats, none was working<br>dropped lines with values out of France range (in gradients) | - |
| `light_condit` | few NA's | minimized groups |
| `location_type` | no NA's | dummy-coded for urban (2) | 
| `intersect_type` | no information to NA | minimized groups |
| `weather_cond` | no information to NA | minimized groups, dummy-coded for normal conditions (1) |
| `collision_type` | no information to NA | minimized groups |
| `municip_code` | given information not straightforward<br> information is better given in `lat` and `long`<br>**dropped** | - |
| `adr` | inconsistent attributes<br>**dropped** | - |
| `gps_code` | just one category and NA<br>**dropped** | - |
| `dep` | 183 unique values with partly low n<br>some NA's<br>inconsistent values<br>information also in `lat`/`long`<br>**dropped** | - |
| `month` | no NAs | cyclical coding: `month_sin`, `month_cos`<br>using `month_sin` and `month_cos`  |
| `day` / `weekday` | no NAs, since information is not consistent (17 is different in march and september) recoded to `weekday` through `date` | cyclical coding: `weekday_sin`, `weekday_cos`<br>using `weekday_sin` and `weekday_cos`  |
| `time` \ `hour` | not consistent format<br>some inconsistent entries such as 43, recoded to 043<br>corrected format<br>extracted hour for new feature `hour`, dropped `time` | cyclical coding: `hour_sin`, `hour_cos` |
| `year` | no NA's  | - |
| `road_cat` | no NA's/no information<br> some categories with low n | minimized groups |
| `road_num` | most values no information or NA<br> information not consistent (sometimes numbers, sometimes street names)<br>numbers are not unique since letter for roads are not indicated: no difference between road N1 and D1<br>**dropped** | - |
| `road_num_index` | over 40% NA's<br>**dropped** | - |
| `v2` | over 40% NA's<br>**dropped** | - |
| `max_speed` | over 40% NA's<br>**dropped** | - | 
| `traffic_dir` | no information to NA's | minimized groups |
| `num_lanes` | not consistent attributes (int, float or string)<br>string, no information recoded to NA<br>all brought to float | minimized groups |
| `pr` | over 40% NA's/no information<br>**dropped** | - |
| `pr1` | over 40% NA's/no information<br>**dropped** | - |
| `res_lane` | ca. 93% has NA or no information<br>**dropped** | - |
| `prof_road` | no information to NA | minimized groups, dummy-coded to flat (1) |
| `plan_view` | no information to NA | minimized groups, dummy-coded to straight (1) |
| `tpc_length` | ca. 91% with NA/no information<br>**dropped** | - |
| `width_road` | ca. 54% with NA/no information<br>inconsistent values (super high values above 120m)<br>information is also given in `num_lanes`<br> **dropped** | - |
| `surface` | 20% with NA or no information<br> a lot of different categories | minimized groups, dummy-coded to normal (1) |
| `infra` | 88% is NA/no information<br>**dropped** | - |
| `acc_loc` | a lot of different groups, some with low n<br> recoded no information to NA  | minimized groups, dummy-coded to on carriageway (1) |

## Dropping variables + cases

I dropped cases, where I don't have information on longitude or lattitude (`NA` or `0`) or couldn't convert into meaningful and interpretable values compared to the information given in gradients in these two features. 

Next I drop the variables I decided above: `municip_code`, `gps_code`, `adr`, `road_num_index`, `v2`, `pr`, `pr1`, `max_speed`, and `road_num`.  

## Feature Selection

During the feature selection, I followed three steps: 1) controlling multi-collinearity, 2) Select K Best, and 3) PCA. 

In the first step, there were no severe violation of multi-collinearity, and I kept a data set with the full data. In the second step, I selected the 10 best features and the PCA created 11 components representing the data that capture 90% of the variance. 

In the base model test, I ran the base models with all three data sets to capture which performs best. 

## Base models

As described above, I opted to go for tree-based models. First, since the relation of the two important geographical features -- lattitude and longitude -- might not be linear. Second, because tree-based models can handle categorical features with a lot of groups more efficiently than linear regression models. Even though I recoded some features to be represented by less groups, most features showed imbalance between categories. These imbalances are again better handled in tree-based models, since they would just be ignorde for making a split. Furthermore, tree-based models are less prone to overfitting in regard to dealing with rare categories.

I used the following basic settings for the base models:

| Model | Setting | 
| ---- | --------- |
| Random Forest |  n_estimators=100<br>random_state=42 |
| XGBoost | n_estimators=100<br>learning_rate=0.1<br>random_state=42 |
| CatBoost | iterations=1000<br>learning_rate=0.05<br>depth=10<br>verbose=200 |

In order to evalute the models and data sets, I calculated *Mean Absolute Error* (MAE), *Mean Squared Error* (MSE), *Root Mean Squared Error* (RMSE), and *$R^2$*. 

To balance the scores, I calculated a mean prediction as baseline model to evaluate the other models' scores. Each model was done in a cross validation with 5 folds and the values below represent the average score.

**Full data set**

| Model | MAE | MSE | RMSE | R2 |
|-------|-----|-----|------|-----|
| Mean Prediction | 1.5521 | 3.4388 | 1.8544 | 0.0000 |
| Random Forest | 1.1392 | 2.4393 | 1.5618 | 0.2906 |
| XGBoost | 1.0966 | 2.3414 | 1.5302 | 0.3191 |
| CatBoost | 1.1069 | 2.3429 | 1.5307 | 0.3186 |

**Select K Best (10 features)**

| Model | MAE | MSE | RMSE | R2 |
|-------|-----|-----|------|-----|
| Mean Prediction | 1.5521 | 3.4388 | 1.8544 | 0.0000 |
| Random Forest | 1.1281 | 2.5738 | 1.5618 | 0.2514 |
| XGBoost | 1.0900 | 2.3289 | 1.5302 | 0.3227 |
| CatBoost | 1.0972 | 2.3334 | 1.5307 | 0.3214 |


**PCA (11 components)**

| Model | MAE | MSE | RMSE | R2 |
|-------|-----|-----|------|-----|
| Mean Prediction | 1.5521 | 3.4388 | 1.8544 | 0.0000 |
| Random Forest | 1.1830 | 2.5471 | 1.5618 | 0.2592 |
| XGBoost | 1.1487 | 2.4891 | 1.5302 | 0.2761 |
| CatBoost | 1.1488 | 2.4570 | 1.5307 | 0.2855 |

As can be seen from the results, models decrease MAE by $0.4-0.5$ and increase $R^2$ to ca. 30% (from 0%). The models with the PCA components performed poorly compared to the full data set and the k-best selection. Therefore, I decided to go further to optimization only with the full data set and the k-best selection. 

In the full data set, Random Forest, XGBoost and CatBoost are quite close in all scores, and, therefore, I decided to go with all three models for optimization. 

In the k-best selection, XGBoost and CatBoost performed better than Random Forest, and, hence, I only take these two models further to optimization. 

## Optimized models

For optimization, I decided to go by grid search. In the grid search, I used cross validation with 5 folds. I tested the following grids:

| Model | Parameter Grid | 
| ----- | ------------- |
| Random Forest | n_estimators: [100, 200, 300]<br>max_depth: [10, 20, 30],<br>min_samples_split: [2, 5, 10],<br>min_samples_leaf: [1, 2, 4] | 
| XGBoost | n_estimators: [100, 200],<br>learning_rate: [0.01, 0.05, 0.1],<br>max_depth: [3, 6, 10],<br>subsample: [0.7, 0.8, 1.0] | 
| CatBoost | iterations: [500, 1000],<br>learning_rate: [0.05, 0.1],<br>depth: [6, 10, 12],<br>l2_leaf_reg: [1, 3, 5]<br> |

As stated above, for the full data set I only optimized XGBoost and CatBoost. The best models are stated below:

**Full data set**

| Model | Best parameters | Best scores |
| ----- | ----------------| -----------| 
| Random Forest | n_estimators: 300<br>max_depth: 20<br>min_samples_split: 10<br>min_samples_leaf: 4| MAE: 1.1136<br>MSE: 2.3682<br>RMSE: 1.5389<br>R2: 0.3113|
| XGBoost | n_estimators: 200<br>learning_rate: 0.1<br>max_depth: 6<br>subsample: 1.0| MAE: 1.0947 <br>MSE: 2.3013<br>RMSE: 1.5170<br>R2: 0.3307 |
| CatBoost | iterations: 1000<br>learning_rate: 0.05<br>depth: 6<br>l2_leaf_reg: 3| MAE: 1.1039 <br>MSE: 2.3029 <br>RMSE: 1.5175<br>R2: 0.3303 |

**k-best selection**

| Model | Best parameters | Best scores |
| ----- | ----------------| -----------| 
| XGBoost | n_estimators: 200 <br>learning_rate: 0.1<br>max_depth: 6<br>subsample: 0.8 | MAE: 1.0903<br>MSE: 2.3053<br>RMSE: 1.5170<br>R2: 0.3296 |
| CatBoost | iterations: 1000<br>learning_rate: 0.1<br>depth: 6<br>l2_leaf_reg: 5| MAE: 1.0922 <br>MSE: 2.2998 <br>RMSE: 1.5175 <br>R2: 0.3312 |

## Final Test Results

For all data, the best model is the XGBoost model. After training on the whole train set, I achieved the following scores for the test set:

| Model | MAE | MSE | RMSE | R2 |
|-------|-----|-----|------|-----|
| XGBoost | 1.0946 | 2.3137 | 1.5211 | 0.3257 | 


For k-selection, the best model is CatBoost. After training on the whole train set, I achieved the following scores for the test set:

| Model | MAE | MSE | RMSE | R2 |
|-------|-----|-----|------|-----|
| CatBoost | 1.0927 | 2.3140 | 1.5212 | 0.3256 | 

