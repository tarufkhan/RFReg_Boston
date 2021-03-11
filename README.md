# Boston house prediction using Random Forest Regressor

Whats done here -
* Import all the required libraries/modules.
* Load the boston house dataset from load_boston() which is in sklearn.datasets.
* Read the dataset using pandas.
* Our dataset has 506 rows and 14 columns.
* Here we don’t have any null value to care about.
* Calculated Mean, Percentile, Standard Deviation of data using describe().
* We used distplot from seaborn on our dataset and got much of a normalized graph. So we don’t need any normalization technique to use.

![img1](/img/rf1.PNG)

* Plotting data for outlier detection.

![img2](/img/rf2.PNG)

* To handle outliers, we used IQR technique, and after that we are left with a dataset shaped 268 rows and 14 columns.
* Again plotting the new data,

![img3](/img/rf3.PNG)

* For Feature selection, we use ols method from statsmodels module.
* During the process we opt various features to calculate r-square and adjusted r-square value. At the end, we decided to drop the ‘ZN’ column from the dataset with r-square and adj. R-square value of 0.71 and 0.70 respectively.
* Checking the Multicollinearity we found that ‘RAD’ and ‘TAX’ have very severe VIF values of 0.9 and 0.5. So we removed ‘RAD’ and got a normal VIF value after that.
* Now, plotted a graph to understand the correlation of independent variables.

![img4](/img/rf4.PNG)

* Now, preparing data for the Random Forest Regressor model. Split data into train and test parts. Fit data on the model.
* Calculating the r-square on training data is 0.96, on testing data it reduces to 0.83. The Adjusted r-square value on training data is 0.96, on testing data it goes down to 0.78.
* The results are not very good so we perform Hyper-Parameter Tuning with the help of GridSearchCV.
* We tuned ‘criterion’ , ‘max_depth’, ‘max_features’, ‘min_samples_leaf’, ‘oob_score’ &‘min_samples_split’.
* After tuning, we got the r-square value of 0.96 on training data and 0.85 on testing data, the adjusted r-square on training data is 0.95 and 0.80 on testing data. This is a bit fine.
* Lastly we performed the regularization to check whether our model is overfitting or not.
* The scores of Lasso, Ridge and ElasticNet are 0.83, 0.82 & 0.83 respectively. Hence we can say that our model is not overfitting.
