"""
BIG DATA ASSIGNMENT 1
Familiarize yourselves with the data and prepare it for the analysis
1. Look at some basic summary statistics
2. Plot your data
3. Pick a classification method (naive Bayes or k Nearest Neighbor) to predict top quality wines
    - Use 10-fold cross validation
4. Assess the performance of your classifier
5. One page summary to a non-technical audience explaining:
    - Description of data method (including key assumptions)
    - Results
    - How to apply as new data comes in
"""
# Part One
# Import packages
import pandas as pd
import numpy as np
import seaborn as sbn; sbn.set()
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, cross_val_score, train_test_split

#Read in data
data_original = pd.read_csv('winequality.csv')
data_original = pd.DataFrame(data_original)
pd.data_original.decribe()

"""
Notes:
- Eleven characteristics used to judge quality
- Class variable is quality (discrete)
- 1597 observations aka bottles
- All features are continuous 
- Average quality is around 5.6, maximum quality is 8 and minimum is 3 (in the sample). A sd of 0.8
- Fixed acidity feature varies a lot (large SD) and volatile acidity does not
"""

# Cleaning data
"""
After running summary statistics, I found that there were two observations that didn't have the quality score. 
I exclude these two observations in my code and subsequent analysis below
"""

# Exclude observations missing quality score
data = data_original[data_original["quality"].notna()]
data = pd.DataFrame(data)
print(data.count(axis=0, level=None, numeric_only=False))
index = data.index


# Part Two: Visualize Data
# Plot distribution of each feature
data.columns = data.columns.astype(str) # Create string of column names

for j in data.columns:
    x = np.array(data[j])
    plt.hist([x])
    plt.title("Distribution of "+j)
    plt.savefig("hist"+"_"+str(j)+".png")
    plt.show()

"""
Notes:
- A few features have obvious outliers:
    - pH (majority <5)
    - Density (majority >0)
    - Chlorides (majority <2)
    - Acidity (majority > -1000)

Outliers (determined to be errors and not just anomalies):
 - pH bottle: level of 30 with a quality score of 6. 
 This is likely a typo considering any bottle with a pH higher than 3.8 is considered unstable wine
 - Chloride bottle: over 11 with a quality score <5. Likely a typo considering most other bottles have a mean of 0-1
 - Density bottle: density of -8g/metric cup. Average is around 1. The bottle scored a quality score of 5, 
 and with a density of 8 it would mean the wine is 800% less dense than water. This was likely a typo
 - Acidity bottles: three observations with acidity under -1000, scored a 5, 6, and 7. 
 Considering the average acidity is 6 grams/liter, these are likely typos and will be removed from the data
 """

# Filter out identified outliers
data_final = data[(data['chlorides'] < 11)]
data_final = pd.DataFrame(data_final)
data_final = data_final[(data_final['density'] > -8)]
data_final = data_final[(data_final['fixed acidity'] > -100)]
data_final = data_final[(data_final['pH'] < 25)]
data_final = pd.DataFrame(data_final)

# Plot each feature against quality
for i in data.columns:
    sbn.boxplot(y=np.array(data[i]), x=np.array(data["quality"]))
    plt.ylabel(str(i))
    plt.xlabel("Quality Score")
    plt.title("Wine quality and " + i)
    plt.savefig("figure"+"_"+str(i)+".png")
    plt.show()

# Distribution of quality
data_final['quality'].value_counts()
import seaborn as sbn; sbn.set()
sbn.countplot(x='quality', data=data_final)
plt.savefig("class_distribution.png")
plt.show()

"""Little observations with a 3,4,7, and 8 quality score. Imbalanced distribution of the class variable
 can lead to an accuracy paradox amongst other things"""

# Make quality binary
data_final['quality_type'] = 'low'
data_final.loc[data_final['quality'] > 5, 'quality_type'] = 'high'
"""736 low quality observations, 852 high quality observations"""

#Transforming Features
"""In order to use Gaussian Naive Bayes classifier we need the features to be normally distributed. 
Features that are skewed include:
- Alcohol
- Citric acid
- Free sulfur dioxide
- Residual sugar
- Total Sulfur dioxide
"""
def sqroot(x):
    return (x**(1/2))

data_final['alcohol'] = data_final['alcohol'].apply([sqroot])
data_final['citric acid'] = data_final['citric acid'].apply([sqroot])
data_final['free sulfur dioxide'] = data_final['free sulfur dioxide'].apply([sqroot])
data_final['residual sugar'] = data_final['residual sugar'].apply([sqroot])
data_final['total sulfur dioxide'] = data_final['total sulfur dioxide'].apply([sqroot])

# Excluding observations with N/A
data_final = data_final.dropna()
data_final = pd.DataFrame(data_final)

# Excluding quality variable (just one class var instead)
data_model = data_final.drop('quality', axis=1)


# Part Three: Classification Method
# Define the naive Bayes Classifier
"""
- Bernoulli Naive Bayes classifier is not appropriate since our features are continuous
- Multinomial Naive Bayes classifier and the Gaussian Bayes classifier are appropriate (given Gaussian distribution of features)
- Cons: the classes aren't well separated (not much difference between a quality of 5 and a quality of 6);
Will avoid the zero frequency problem by using 10 fold cross validation when assessing accuracy of both models
"""

# Create the X, Y, Training and Test
msk = np.random.rand(len(data_model)) < 0.9
train = data_model[msk]
test = data_model[~msk]
# Training data
xtrain = train.drop('quality_type', axis=1)
ytrain = train.loc[:, 'quality_type']
# Test data
xtest = test.drop('quality_type', axis=1)
ytest = test.loc[:, 'quality_type']

# Testing various models
classifier_models = [MultinomialNB(), GaussianNB()]
for i in classifier_models:
    model = i
    classifier = model.fit(xtrain, ytrain) # Train the model
    pred = model.predict(xtest) # Predict output using the test data
    mat = confusion_matrix(pred, ytest) # Plot Confusion Matrix
    names = np.unique(pred)
    sbn.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
             xticklabels=names, yticklabels=names)
    plt.xlabel('Truth')
    plt.ylabel('Predicted')
    plt.savefig("confusionmatrix.png")
    plt.show()
    accuracies=cross_val_score(classifier,X=xtrain,y=ytrain,cv=10) #Calculate accuracy using cross validation
    print("Accuracy:{:.2f}%".format(accuracies.mean()*100))
    print(accuracies)

# Printing confusion matrix (for superior model)
model_G = GaussianNB()
classifier = model_G.fit(xtrain, ytrain) # Train the model
pred = model_G.predict(xtest) # Predict output using the test data
mat = confusion_matrix(pred, ytest) # Plot Confusion Matrix
names = np.unique(pred)
sbn.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
        xticklabels=names, yticklabels=names)
plt.xlabel('Truth')
plt.ylabel('Predicted')
plt.savefig("Gaussian_confusionmatrix.png")
plt.show()
