#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: notebook.ipynb
Conversion Date: 2025-09-09T10:03:04.068Z
"""

"""This notebook explores customer churn prediction using a structured analysis of the dataset. It includes data exploration, 
feature analysis, and statistical methods to identify key predictors of churn. Following the analysis, various machine learning models 
are trained and evaluated to predict customer retention."""

# importing libraries
import numpy as np
import pandas as pd
pd.options.display.max_columns= None
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import scipy.stats as ss

# reading the data
df = pd.read_csv('/kaggle/input/bank-customer-churn/Customer-Churn-Records.csv')
data = df.copy()

print(f'shape of the dataframe is {data.shape}')

data.columns.to_series().groupby(data.dtypes).groups   

data.info() # there is no missing values 

data[data.duplicated()]
# we have no duplicated rows

"""from previous explortion, the data has no missing values, no duplicates and of shape (10000, 18).there are 3 columns that do not contrbute
to the problem of customer churn rate which are (RowNumber, CustomerId, Surname)."""

data.describe()

# plotting the histograms for integers and floating columns
data.hist(figsize=(20, 24))
plt.show()

# investigating object columns
obje_columns = ['Geography', 'Gender', 'Card Type']

for column in obje_columns:
    if column in data.columns:
        print(f"Value counts for {column}:")
        print(data[column].value_counts())
        print("\n")
        
        plt.figure(figsize=(10, 6))
        data[column].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title(f"Bar plot of {column}")
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)  
        plt.show()
    else:
        print(f"Column {column} does not exist in the DataFrame.")

"""from features distributions we can see that we have 10 categorical features whether they are integer or objects and 5 continous features +
the 3 mentioned features that do not contribute to the problem for the object categorical features, the data is almost equal in terms of 
gender(male, female) and card types(diamond, gold, silver, platinum)"""

# Investigating the target variable
churn_counts = data['Exited'].value_counts()
churn_percentages = (churn_counts / len(data)) * 100

print("Number of customers by churn status:")
print(churn_counts)
print("\nPercentage of customers by churn status:")
for index, percentage in churn_percentages.items():
    print(f"{index}: {percentage:.2f}%")

# dropping unnecessatry features
data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)
data.shape

# We will split the data into continous features and categorical features (object and integers) Investigating the continuous features only
con_features = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary', 'Point Earned']
data[con_features].describe()

# note that the minimum and the first quartile for Balance feature are zeros which will lead us later to further analysis

"""From previous results we can conclude that the continous features has no outliers but we will check them 
   further using modified Z-score.
   
   We will only check Age, CreditScore and Balance columns because the EstimatedSalary and Point Earned is almost 
   uniform feature which have no outliers as there is no central tendency of the data so the mean of a uniform 
   distribution is not the best guess to the value of the next observation, in the following links this idea of 
   uniform distribution will be explained clearly 
   'https://www.researchgate.net/post/Why-does-Uniform-Distributions-have-no-outliers'
   'https://en.wikipedia.org/wiki/Kurtosis'   
   
    For checking for outliers in the other three columns, we will use 'Median Absolute Deviation' because the 
    Balance feature is 'Bimodal Distribution' has a significant peak with a large proportion of values 
    concentrated around 0, and a more normal distribution beyond this range. 
    
    The standard deviation is sensitive to this skew and the long tail. Therefore,the Median Absolute Deviation is used as it is 
    more robust to outliers and provides a clearer measure of dispersion for Bimodal distributions
    integrated into the modified Z-score version.
    
    Although the summary statistics for Age and CreditScore seems to be reasonable and they have no outliers
    but we will check them too"""


def modified_z_score(series):
    median = series.median()
    mad = ss.median_abs_deviation(series, scale='normal')
    return 0.6745 * (series - median) / mad


for feature in ['CreditScore', 'Age', 'Balance']:
    z_scores = modified_z_score(data[feature])
    outliers = data[abs(z_scores) > 3.5]
    
    print(f'Feature: {feature}')
    print(f'Number of Outliers: {outliers.shape[0]}')
    print(f'Percentage of Outliers: {100 * outliers.shape[0] / data.shape[0]:.2f}%')
    print()
# the results show no outliers to the data

# we will check the normality for each of the 3 columns using q-q plot 
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

ss.probplot(data['Age'], dist="norm", plot=axs[0])
axs[0].set_title('Q-Q Plot of Age')

ss.probplot(data['CreditScore'], dist="norm", plot=axs[1])
axs[1].set_title('Q-Q Plot of CreditScore')

ss.probplot(data['Balance'], dist="norm", plot=axs[2])
axs[2].set_title('Q-Q Plot of Balance')

plt.tight_layout()
plt.show()


"""We can observe here that:
     1- CreditScore is almost a perfect normal distributionwith a very slight negatively skewness 
     2- Age feature is closer to postive skew distribution than normal distribution
     3- There is an issue in the Balance feature because of the high left peak
   
   We will investigate the Age and Balance features further regarding their relationship with the target"""

# investigating the non-continuos features
all_features = data.columns.tolist()

cat_features = [feature for feature in all_features if feature not in con_features]
print(cat_features, len(cat_features))

#plotting the unique values percentage and the percentage of leavers for each value
for feature in cat_features:
    value_counts = (data[feature].value_counts(normalize=True)) * 100  
    print(f"\nPercentage of unique categories for {feature}:")
    print(value_counts)
          
    leavers_pct = data.groupby(feature)['Exited'].mean() * 100
    leavers_pct = leavers_pct.reindex(value_counts.index, fill_value=0)
    print(f"\nPercentage of leavers by {feature}:")
    print(leavers_pct)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6), sharey=False)
    
    ax_bar = axes[0]
    
    # Calculate percentages manually
    counts = data[feature].value_counts(normalize=True).mul(100).reindex(value_counts.index)
    sns.barplot(x=counts.index, y=counts.values, ax=ax_bar, color='steelblue') # use sns.countplot with stat argument = 'percent' if you use 0.13 version or later 
    ax_bar.set_title(f'Distribution of {feature} percentage')
    ax_bar.set_xlabel(feature)
    ax_bar.set_ylabel('Percentage')
    ax_bar.grid(True)
    
    leavers_pct.plot(kind='bar', ax=axes[1], color='indianred')
    axes[1].set_title(f'Percentage of Leavers by {feature} classes')
    axes[1].set_xlabel(feature)
    axes[1].set_ylabel('Percentage')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()


"""We can see that some features are really impactful for churn rate as following:
    1- Geography: the percentage of leavers in Germany is double of the percentage in Spain and France.
    2- Gender: the leaving females are more than leaving males by significant percentage.
    3- NumOfProducts: this feature is a little bit tricky as we can see all customer with four purchased products
       and about 80% of customers with three purchased products leave the bank but actually those customers are 
       only about 3% of the total dataset and the rest is customers with only one purchased products and 
       two purchased products and we can also observe that the percentage of leavers with only one product is 
       three times of percentage leavers with two products which means that about 70% of leavers only 
       purchase one product.
    4- IsActiveMember: percentage of leavers from non active members are double of the active members.
    5- Complain: customers that complain leaves the bank by 99.5%.
    6- the other four columns have no significant impact on the churn rate."""

# we will now investigating the continous features through plotting the histogrgram and KDE for both existing and churn customers
fig, axes = plt.subplots(len(con_features), 2, figsize=(15, 5 * len(con_features)), sharex=False, sharey=False)

for i, feature in enumerate(con_features):
    ax_hist = axes[i, 0]
    ax_kde = axes[i, 1]
    
    sns.histplot(data[feature], kde=False, color='gray', ax=ax_hist, bins=60, label='Overall')
    ax_hist.set_title(f'Histogram of {feature}')
    ax_hist.set_xlabel(feature)
    ax_hist.set_ylabel('Frequency')
    
    sns.kdeplot(data.loc[data['Exited'] == 1, feature], color='red', ax=ax_kde, label='Churn Customers')
    sns.kdeplot(data.loc[data['Exited'] == 0, feature], color='green', ax=ax_kde, label='Existing Customers')
    ax_kde.set_title(f'KDE of {feature}')
    ax_kde.set_xlabel(feature)
    ax_kde.set_ylabel('Density')
    ax_kde.legend()
    
    ax_kde.set_xlim(left=data[feature].min(), right=data[feature].max())

plt.tight_layout()
plt.show()



"""Here is what I can observe from the figure 
     1- Both ['Point Earned', 'EstimatedSalary'] follow uniform distribution as mentioned above and their KDEs 
        are nearly identical for both churned and existing customers which indicates that those features do not 
        significantly differentiate between the two customer groups.
     
     2- For 'CreditScore', one can observe that the data follows a normal distribution as clarified in Q-Q plot
        but there is a large portion of the customers at the most right of the distribution which will be 
        further analyzed numerically regarding the distribution of churn rate at this portion but at the first
        look the KDEs for this feature indicates there is no differnce between them.
     
     3- For 'Age' feature, it is obvious that older customer are more likely to churn according to the KDEs
        also the distribution of the 'Age' is positively skewed, A transformation will be applied to this feature.
     
     4- For the 'Balance' I will consider checking this large portion at the left regarding its relationship 
        with the target as this is the only part of the distribution that show a higher estimation for existing
        customers over churned customers."""

# further analysis for the CreditScore 
high_credit_score_data = data[data['CreditScore'] > 840]

plt.figure(figsize=(10, 6))
sns.kdeplot(high_credit_score_data.loc[high_credit_score_data['Exited'] == 1, 'CreditScore'], label='Churn Customers', color='blue')
sns.kdeplot(high_credit_score_data.loc[high_credit_score_data['Exited'] == 0, 'CreditScore'], label='Existing Customers', color='green')

plt.title('KDE of CreditScore > 840 by Customer Type')
plt.xlabel('CreditScore')
plt.ylabel('Density')
plt.legend()
plt.show()

print(f"The skewness fot the CreditScore is {data['CreditScore'].skew()}")

value_counts = high_credit_score_data['Exited'].value_counts()
print("Value Counts for customers with CreditScore > 840:")
print(value_counts)

churn_percentage = value_counts[1] / value_counts.sum() * 100
print(f'Percentage of churned customers in this group: {churn_percentage:.2f}%')


"""The results for the further analysis as following:
    1- The KDE for existing customers is higher than for churned ones in this group which makes sense for the 
       portion with the highest CreditScore. 
    2- The skewness of the `CreditScore` feature is very close to zero confirming that it is almost symmetric.
    3- The percentage of churned customers in this subset is 19.93% which is consistent with the overall 
       churn rate of 20.38% in the entire dataset."""

# next we will apply several transformations for Age feature and further analyze it
data['Age_LogTransformed'] = np.log(data.Age)
data['Age_BoxCoxTransformed'], parameter = ss.boxcox(data.Age)

age_features = ['Age', 'Age_LogTransformed', 'Age_BoxCoxTransformed']

for feature in age_features:
    if feature == 'Age_BoxCoxTransformed':
        print(f'The skewness of the {feature} feature is {data[feature].skew()} and parameter is {parameter}')
    else:
        print(f'The skewness of the {feature} feature is {data[feature].skew()}')


        
"""Now we have transformed our data using BoxCOX and natural logarithm tranformations, we can see that BoxCox 
   is the best choice to transform the data to almost has a standard normality
   
   Note that when the parameter of BoxCox is equal to 0, it is actually a natural logarithm that we 
   also applied so the first transformation is a special case of the second transformation"""

# plotting the histogram for the tranformed data and the KDEs for both classes (esisting, churn)  
age_features = ['Age', 'Age_LogTransformed', 'Age_BoxCoxTransformed']

fig, axes = plt.subplots(len(age_features), 2, figsize=(15, 5 * len(age_features)), sharex=False, sharey=False)

for i, feature in enumerate(age_features):
    ax_hist = axes[i, 0]
    ax_kde = axes[i, 1]
    
    sns.histplot(data[feature], kde=False, color='gray', ax=ax_hist, bins=60, label='Overall')
    ax_hist.set_title(f'Histogram of {feature}')
    ax_hist.set_xlabel(feature)
    ax_hist.set_ylabel('Frequency')
    
    sns.kdeplot(data.loc[data['Exited'] == 1, feature], color='red', ax=ax_kde, label='Churn Customers')
    sns.kdeplot(data.loc[data['Exited'] == 0, feature], color='green', ax=ax_kde, label='Existing Customers')
    ax_kde.set_title(f'KDE of {feature}')
    ax_kde.set_xlabel(feature)
    ax_kde.set_ylabel('Density')
    ax_kde.legend()
    
    ax_kde.set_xlim(left=data[feature].min(), right=data[feature].max())

plt.tight_layout()
plt.show()


"""Plotting the histogram shows that the transformed data are more closer to the standard normal distribution
   where the KDEs still indicate the same as the original data which is that the older the customers the more 
   lokely they churn."""

# plotting the q-q plot for the new transformed data for Age feature
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

ss.probplot(data['Age'], dist="norm", plot=axs[0])
axs[0].set_title('Q-Q Plot of Age')

ss.probplot(data['Age_LogTransformed'], dist="norm", plot=axs[1])
axs[1].set_title('Q-Q Plot of Age_LogTransformed')

ss.probplot(data['Age_BoxCoxTransformed'], dist="norm", plot=axs[2])
axs[2].set_title('Q-Q Plot of Age_BoxCoxTransformed')

plt.tight_layout()
plt.show()

# now we will consider the Balance feature and analyze it regarding the large portion of the data around zero
# dividing the feature in ranges to get the customers in each range by thresholds
thresholds = [0, 50000, 100000, 150000, 200000, 250000]
total_customers = data.shape[0]

zero_balance_count = data[data['Balance'] == 0].shape[0]
zero_balance_percentage = (zero_balance_count / total_customers) * 100
print(f'Number of customers with a balance of 0: {zero_balance_count} ({zero_balance_percentage:.2f}%)')

highest_balance_count = data[data['Balance'] > 250000].shape[0]
highest_balance_percentage = (highest_balance_count / total_customers) * 100
print(f'Number of customers with a balance over 250000: {highest_balance_count} ({highest_balance_percentage:.2f}%)')

for i in range(len(thresholds) - 1):
    lower_threshold = thresholds[i]
    upper_threshold = thresholds[i + 1]
    
    count = data[(data['Balance'] > lower_threshold) & (data['Balance'] <= upper_threshold)].shape[0]
    percentage = (count / total_customers) * 100
    print(f'Number of customers within the range {lower_threshold} and {upper_threshold}: {count} ({percentage:.2f}%)')



"""no we see that we have over 3600 cutomers with balance equal to 0 balance, we need to investigate 
   this large percentage and analyze its relationship with other features"""

# this is a sanity check to make sure the 75 between 0 and 50k do not have something not normal
low_balance = data[(data['Balance'] > 0) & (data['Balance'] <= 50000)]

low_churn_count = low_balance[low_balance['Exited'] == 1].shape[0]
low_total_count = low_balance.shape[0]
low_bal_churn_perc = low_churn_count / low_total_count

print(f'Total number of customers with low balance: {low_total_count}')
print(f'Number of churned customers with low balance: {low_churn_count}')
print(f'Number of non-churned customers with low balance: {low_total_count - low_churn_count}')

print(f'Percentage of churned customers with low balance {low_bal_churn_perc * 100:.2f}%')


"""the results for this portion is quite interesting, despite the number of customers in this portion is less
   1% but their churn rate is significantly high (34.67%) compared to the overall churn rate (20.38%)
   now we will do the same for the other 2 classes (customers with 0 balance and customers with > 50k)"""

zerp_balance = data[data['Balance'] == 0]

zero_churn_count = zerp_balance[zerp_balance['Exited'] == 1].shape[0]
zero_total_count = zerp_balance.shape[0]
zero_bal_churn_perc = zero_churn_count / zero_total_count

print(f'Total number of customers with zero balance: {zero_total_count}')
print(f'Number of churned customers with zero balance: {zero_churn_count}')
print(f'Number of non-churned customers with zero balance: {zero_total_count - zero_churn_count}')

print(f'Percentage of churned customers with zero balance {zero_bal_churn_perc * 100:.2f}%')


"""Despite the customers with 0 balance are the largest portion of the total data but the churn rate for this 
   portion is somehow small (13.82%) compared to the overall churn rate 20.38%"""

high_balance = data[data['Balance'] > 50000]

high_churn_count = high_balance[high_balance['Exited'] == 1].shape[0]
high_total_count = high_balance.shape[0]
high_bal_churn_perc = high_churn_count / high_total_count

print(f'Total number of customers with high balance: {high_total_count}')
print(f'Number of churned customers with high balance: {high_churn_count}')
print(f'Number of non-churned customers with high balance: {high_total_count - high_churn_count}')

print(f'Percentage of churned customers with high balance {high_bal_churn_perc * 100:.2f}%')

#now let's plot the hist and KDEs for the portion of the customers that have balance over 50K and its skewness
print(f"The skewness for the Balnce feature above 50K is {high_balance['Balance'].skew()}")

fig, axs = plt.subplots(1, 3, figsize=(24, 8))

sns.histplot(high_balance['Balance'], kde=False, color='gray', ax=axs[0], bins=60, label='Overall')
axs[0].set_title('Histogram of High Balance Customers', fontsize=16)
axs[0].set_xlabel('Balance', fontsize=14)
axs[0].set_ylabel('Frequency', fontsize=14)
axs[0].grid(True)

ss.probplot(high_balance['Balance'], dist="norm", plot=axs[1])
axs[1].set_title('Q-Q Plot of High Balances', fontsize=16)

sns.kdeplot(high_balance.loc[high_balance['Exited'] == 1, 'Balance'], color='red', label='Churn Customers', ax=axs[2])
sns.kdeplot(high_balance.loc[high_balance['Exited'] == 0, 'Balance'], color='green', label='Existing Customers', ax=axs[2])
axs[2].set_title('KDE of High Balance', fontsize=16)
axs[2].set_xlabel('Balance', fontsize=14)
axs[2].set_ylabel('Density', fontsize=14)
axs[2].legend(fontsize=12)
axs[2].grid(True)

plt.tight_layout()
plt.show()


"""Here we can observe that he Balane data above 50K is almost normal, we can also observe that the estimation 
   for churn customers between 100K and nearly 140K is higher than estimation for existing customers which are
   the large portion of the customers about 38% of total dataset, this insight was obvious in the KDEs for the 
   whole Balance data rather than this we can see the estimation for both churned and existing customers 
   nearly the same (in the KDEs for the whole Balance data the estimation for churn rate dominated the estimation
   for existing customers except for customers with zero balance)"""

def categorize_balance(balance):
    if balance <= 0:
        return 'Zero'
    elif balance <= 50000:
        return '0-50K'
    else:
        return '50K+'

data['Balance_Category'] = data['Balance'].apply(categorize_balance)

zero_count = (data['Balance_Category'] == 'Zero').sum()
print(zero_count)


Balance_cot_features = ['Age', 'Age_BoxCoxTransformed', 'Age_LogTransformed', 'CreditScore', \
                        'EstimatedSalary', 'Point Earned']  
balance_categories = data['Balance_Category'].unique()

n_features = len(Balance_cot_features)
n_categories = len(balance_categories)
fig, axes = plt.subplots(n_features, n_categories, figsize=(5 * n_categories, 4 * n_features))

for i, feature in enumerate(Balance_cot_features):
    for j, category in enumerate(balance_categories):
        ax = axes[i, j] if n_features > 1 else axes[j]
        sns.histplot(data=data[data['Balance_Category'] == category], x=feature, bins=30, ax=ax, color='skyblue')
        ax.set_title(f'{feature} - Balance: {category}')
        if i == n_features - 1:
            ax.set_xlabel(feature)
        if j == 0:
            ax.set_ylabel('Count')

plt.suptitle('Distribution of Features by Balance Category', y=1)
plt.tight_layout()
plt.show()

Balance_cat_features = ['Geography', 'Gender', 'NumOfProducts', 'IsActiveMember', 'Complain', 
                        'HasCrCard', 'Tenure', 'Satisfaction Score', 'Card Type']

target_feature = 'Balance_Category'

for feature in Balance_cat_features:
    grouped_df = data.groupby([target_feature, feature]).size().reset_index(name='count')
    total_counts = grouped_df.groupby(target_feature)['count'].sum().reset_index(name='total_count')
    
    merged_df = pd.merge(grouped_df, total_counts, on=target_feature)
    merged_df['percentage'] = (merged_df['count'] / merged_df['total_count']) * 100
    
    print(f"\n{'='*50}")
    print(f"Percentage distribution for feature: {feature}")
    print(f"{'='*50}\n")
    
    for idx, row in merged_df.iterrows():
        print(f"{feature}: {row[feature]} | {target_feature}: {row[target_feature]} | Percentage: {row['percentage']:.2f}%")
    
    pivot_df = merged_df.pivot(index=feature, columns=target_feature, values='percentage')
    
    pivot_df.plot(kind='bar',figsize=(12, 8), colormap='turbo')
    plt.title(f'Percentage Distribution of {target_feature} by {feature}', y=1.05)
    plt.xlabel(feature)
    plt.ylabel('Percentage')
    plt.legend(title=target_feature)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


data.columns

trans_data = data.drop(['Age_LogTransformed', 'Age', 'Balance', 'Complain'], axis=1)

"""The "Complain" feature was dropped due to its near-perfect correlation (correlation coefficient ~ 1)
 with the target variable. Including it would lead to data leakage, as it directly reveals the outcome,
 causing the model to overfit and not generalize well to new data. Removing this feature ensures that 
 the model learns from other features and remains robust in real-world scenarios."""


print(trans_data.shape, trans_data.columns)

trans_data['Balance_Category'] = trans_data['Balance_Category'].astype('object')

trans_data.info()

# Encoding the object features
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

obj_features = ['Geography', 'Gender', 'Card Type', 'Balance_Category']
label_encoder = LabelEncoder()

for col in obj_features:
    unique_values = trans_data[col].nunique()

    if unique_values == 2:
        trans_data[col] = label_encoder.fit_transform(trans_data[col])
    else:
        trans_data = pd.get_dummies(trans_data, columns=[col])

trans_data.head()


#splitting the data
from sklearn.model_selection import train_test_split

X = trans_data.drop('Exited', axis=1)  
y = trans_data['Exited']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

#scalling the data
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()

X_train_transformed = scaler.fit_transform(X_train)
X_test_transformed = scaler.transform(X_test)

#fitting 4 models as base ones(Benchmarks)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Initialize the models
logistic_model = LogisticRegression(random_state=42)
random_forest_model = RandomForestClassifier(random_state=42)
svm_model = SVC(probability=True, random_state=42)  
xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

# Fit the models on the training data
logistic_model.fit(X_train_transformed, y_train)
random_forest_model.fit(X_train_transformed, y_train)
svm_model.fit(X_train_transformed, y_train)
xgb_model.fit(X_train_transformed, y_train)


# Predict on the test set
logistic_preds = logistic_model.predict(X_test_transformed)
random_forest_preds = random_forest_model.predict(X_test_transformed)
svm_preds = svm_model.predict(X_test_transformed)
xgb_preds = xgb_model.predict(X_test_transformed)

# We will be using Precision-Recall curve and score as they are more informative for imbalanced datasets like churn prediction, 
# they focus on the performance of the model with respect to the minority class (churned customers).

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_curve, average_precision_score

# Evaluate Logistic Regression
print("Logistic Regression Performance:")
print("Accuracy:", accuracy_score(y_test, logistic_preds))
print("Classification Report:\n", classification_report(y_test, logistic_preds))

# Evaluate Random Forest
print("\nRandom Forest Performance:")
print("Accuracy:", accuracy_score(y_test, random_forest_preds))
print("Classification Report:\n", classification_report(y_test, random_forest_preds))


# Evaluate Support Vector Machines
print("Support Vector Machines Performance:")
print("Accuracy:", accuracy_score(y_test, svm_preds))
print("Classification Report:\n", classification_report(y_test, svm_preds))

# Evaluate XGBoost
print("\nXGBoost Performance:")
print("Accuracy:", accuracy_score(y_test, xgb_preds))
print("Classification Report:\n", classification_report(y_test, xgb_preds))

# Confusion matrices for each model
logistic_cm = confusion_matrix(y_test, logistic_preds)
random_forest_cm = confusion_matrix(y_test, random_forest_preds)
svm_cm = confusion_matrix(y_test, svm_preds)
xgboost_cm = confusion_matrix(y_test, xgb_preds)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

sns.heatmap(logistic_cm, annot=True, fmt='d', cmap='Reds', cbar=False, ax=axes[0, 0])
axes[0, 0].set_title('Logistic Regression Confusion Matrix')
axes[0, 0].set_xlabel('Predicted Label')
axes[0, 0].set_ylabel('True Label')

sns.heatmap(random_forest_cm, annot=True, fmt='d', cmap='Greens', cbar=False, ax=axes[0, 1])
axes[0, 1].set_title('Random Forest Confusion Matrix')
axes[0, 1].set_xlabel('Predicted Label')
axes[0, 1].set_ylabel('True Label')

sns.heatmap(svm_cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[1, 0])
axes[1, 0].set_title('Support Vector Machines Confusion Matrix')
axes[1, 0].set_xlabel('Predicted Label')
axes[1, 0].set_ylabel('True Label')

sns.heatmap(xgboost_cm, annot=True, fmt='d', cmap='Purples', cbar=False, ax=axes[1, 1])
axes[1, 1].set_title('XGBoost Confusion Matrix')
axes[1, 1].set_xlabel('Predicted Label')
axes[1, 1].set_ylabel('True Label')

plt.tight_layout()
plt.show()


# Calculate Precision-Recall curve and Average Precision for each model
logistic_probs = logistic_model.predict_proba(X_test_transformed)[:, 1] 
logistic_precision, logistic_recall, _ = precision_recall_curve(y_test, logistic_probs)
logistic_ap = average_precision_score(y_test, logistic_probs, average= 'weighted')
print(f'Average Precision for Logistic Regression model is {logistic_ap:.2f}')

random_forest_probs = random_forest_model.predict_proba(X_test_transformed)[:, 1]  
random_forest_precision, random_forest_recall, _ = precision_recall_curve(y_test, random_forest_probs)
random_forest_ap = average_precision_score(y_test, random_forest_probs, average= 'weighted')
print(f'Average Precision for Random Forest model is {random_forest_ap:.2f}')

svm_probs = svm_model.predict_proba(X_test_transformed)[:, 1] 
svm_precision, svm_recall, _ = precision_recall_curve(y_test, svm_probs)
svm_ap = average_precision_score(y_test, svm_probs, average= 'weighted')
print(f'Average Precision for Support Vector Machine model is {svm_ap:.2f}')

xgb_probs = xgb_model.predict_proba(X_test_transformed)[:, 1]
xgb_precision, xgb_recall, _ = precision_recall_curve(y_test, xgb_probs)
xgb_ap = average_precision_score(y_test, xgb_probs, average= 'weighted')
print(f'Average Precision for XGBoost model is {xgb_ap:.2f}')

# Plot all Precision-Recall curves in a single figure
plt.figure(figsize=(10, 8))

plt.plot(logistic_recall, logistic_precision, label=f'Logistic Regression (AP = {logistic_ap:.2f})', color='red')
plt.plot(random_forest_recall, random_forest_precision, label=f'Random Forest (AP = {random_forest_ap:.2f})', color='darkgreen')
plt.plot(svm_recall, svm_precision, label=f'Support Vector Machine (AP = {svm_ap:.2f})', color='blue')
plt.plot(xgb_recall, xgb_precision, label=f'XGBoost (AP = {xgb_ap:.2f})', color='darkorange')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves for Different Models')
plt.legend()
plt.show()


# As the results indicate that SVM, RF and XGB give the better results compared to lr, we will cintinue fine tuning them 
from sklearn.model_selection import GridSearchCV

# Define SVM model and parameters for GridSearch
svm_model = SVC(probability=True, random_state=42)  
svm_params = {
    'C': [0.1, 1, 10],
    'kernel': ['poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto'],
    'class_weight':[{0:1,1:2}, {0:1,1:4}, {0:1,1:6}]
}

# GridSearchCV for SVM
svm_grid_search = GridSearchCV(estimator=svm_model, param_grid=svm_params, cv=10, n_jobs=-1, verbose=2, scoring='average_precision')
svm_grid_search.fit(X_train_transformed, y_train)

# Best XGBoost Model
svm_grid_model = svm_grid_search.best_estimator_

# Define Random Forest model and parameters for GridSearch
rf_model = RandomForestClassifier(random_state=42)
rf_params = {
    'n_estimators': [100, 200, 300],
    'criterion': ['gini', 'entropy'],
    'max_depth': [10, 20, 30],
    'class_weight': [{0: 1, 1: 2}, {0: 1, 1: 4}, {0: 1, 1: 6}]
}

# GridSearchCV for Random Forest
rf_grid_search = GridSearchCV(estimator=rf_model, param_grid=rf_params, cv=10, n_jobs=-1, verbose=2, scoring='average_precision')
rf_grid_search.fit(X_train_transformed, y_train)

# Best Random Forest Model
best_rf_model = rf_grid_search.best_estimator_


# Define XGBoost model and parameters for GridSearch
xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', objective= 'binary:logistic')
xgb_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 6, 8],
    'learning_rate': [0.05, 0.1, 0.3],
    'scale_pos_weight': [2, 4, 6]
}

# GridSearchCV for XGBoost
xgb_grid_search = GridSearchCV(estimator=xgb_model, param_grid=xgb_params, cv=10, n_jobs=-1, verbose=2, scoring='average_precision')
xgb_grid_search.fit(X_train_transformed, y_train)

# Best XGBoost Model
best_xgb_model = xgb_grid_search.best_estimator_

# Predictions and Evaluation for Support Vector Machines
svm_test_score = svm_grid_model.score(X_test_transformed, y_test)

print("Support Vector Machines Performance:")
print(f"Best Params: {svm_grid_search.best_params_}")
print(f"Best Score (Cross-Validation): {svm_grid_search.best_score_}")
print("Test Score:", svm_test_score)
print("Test Classification Report:")
print(classification_report(y_test, svm_grid_model.predict(X_test_transformed)))


# Predictions and Evaluation for Random Forest
rf_test_score = best_rf_model.score(X_test_transformed, y_test)

print("Random Forest Performance:")
print(f"Best Params: {rf_grid_search.best_params_}")
print(f"Best Score (Cross-Validation): {rf_grid_search.best_score_}")
print("Test Score:", rf_test_score)
print("Test Classification Report:")
print(classification_report(y_test, best_rf_model.predict(X_test_transformed)))


# Predictions and Evaluation for XGBoost
xgb_test_score = best_xgb_model.score(X_test_transformed, y_test)

print("XGBoost Performance:")
print(f"Best Params: {xgb_grid_search.best_params_}")
print(f"Best Score (Cross-Validation): {xgb_grid_search.best_score_}")
print("Test Score:", xgb_test_score)
print("Test Classification Report:")
print(classification_report(y_test, best_xgb_model.predict(X_test_transformed)))


# Calculate Precision-Recall curve and Average Precision for each model

# Support Vector Machine
svm_probs = svm_grid_model.predict_proba(X_test_transformed)[:, 1]
svm_precision, svm_recall, _ = precision_recall_curve(y_test, svm_probs)
svm_ap = average_precision_score(y_test, svm_probs, average='weighted')
print(f'Average Precision for Support Vector Machine model is {svm_ap:.2f}')

# Random Forest
rf_probs = best_rf_model.predict_proba(X_test_transformed)[:, 1]
rf_precision, rf_recall, _ = precision_recall_curve(y_test, rf_probs)
rf_ap = average_precision_score(y_test, rf_probs, average='weighted')
print(f'Average Precision for Random Forest model is {rf_ap:.2f}')

# XGBoost
xgb_probs = best_xgb_model.predict_proba(X_test_transformed)[:, 1]
xgb_precision, xgb_recall, _ = precision_recall_curve(y_test, xgb_probs)
xgb_ap = average_precision_score(y_test, xgb_probs, average='weighted')
print(f'Average Precision for XGBoost model is {xgb_ap:.2f}')

# Plot all Precision-Recall curves in a single figure
plt.figure(figsize=(10, 8))

plt.plot(svm_recall, svm_precision, label=f'Support Vector Machine (AP = {svm_ap:.2f})', color='blue')
plt.plot(rf_recall, rf_precision, label=f'Random Forest (AP = {rf_ap:.2f})', color='darkgreen')
plt.plot(xgb_recall, xgb_precision, label=f'XGBoost (AP = {xgb_ap:.2f})', color='darkorange')

# Add plot labels and title
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves for Different Models')
plt.legend()
plt.show()


# Predictions for each model
svm_preds = svm_grid_model.predict(X_test_transformed)
rf_preds = best_rf_model.predict(X_test_transformed)
xgb_preds = best_xgb_model.predict(X_test_transformed)

# Confusion matrices for each model
svm_cm = confusion_matrix(y_test, svm_preds)
rf_cm = confusion_matrix(y_test, rf_preds)
xgb_cm = confusion_matrix(y_test, xgb_preds)

# Plot confusion matrices in a 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# SVM Confusion Matrix
sns.heatmap(svm_cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[0, 0])
axes[0, 0].set_title('Support Vector Machine Confusion Matrix')
axes[0, 0].set_xlabel('Predicted Label')
axes[0, 0].set_ylabel('True Label')

# Random Forest Confusion Matrix
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Greens', cbar=False, ax=axes[0, 1])
axes[0, 1].set_title('Random Forest Confusion Matrix')
axes[0, 1].set_xlabel('Predicted Label')
axes[0, 1].set_ylabel('True Label')

# XGBoost Confusion Matrix
sns.heatmap(xgb_cm, annot=True, fmt='d', cmap='Purples', cbar=False, ax=axes[1, 0])
axes[1, 0].set_title('XGBoost Confusion Matrix')
axes[1, 0].set_xlabel('Predicted Label')
axes[1, 0].set_ylabel('True Label')

# Adjust layout
plt.tight_layout()
plt.show()