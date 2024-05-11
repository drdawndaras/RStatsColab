---
  title: "CreditCardFraud"
author: "Dawn Daras MS"
date: "`r Sys.Date()`"
output: html_document
---
  
  ```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## Credit Card Fraud Demo by Dawn Daras MS


##This is a demo in RStats of predicting credit card fraud in a dataset.  We will be going through the outlined roadmap below, which includes several steps.

##Of note, we are using a publicly available, anonymized dataset "creditcard.csv."  


#The dataset has already gone through PCA (Principle Component Analysis).  I have already posted a tutorial on PCA, which you can follow.  Here we will discuss how it impacts the analysis.  Additionally, the target, Independent Variable (IV) is labeled "Class," and is the Fraud target.  It is already labeled in the set as '0' or '1.' We will discuss how this would be deployed in an enterprise environment.

##Roadmap
##1) EDA's - Exploratory Data Analysis of our data
##2) Preprocessing/Cleaning the Data
##3) Splitting the data into train/test
##4) Sampling/Under-Sampling
##5) Correlation Matrices
##6) Contrasting Decision Trees with Logistic Regression

##References: 

##Analytics Vidya (2024) "Practical Guide for Dealing with Imbalanced Classification Problems in R" [Analytics Vidya](https://www.analyticsvidhya.com/blog/2016/03/practical-guide-deal-imbalanced-classification-problems/)

##Daras, D. (2024)"PCA R FakeBills" Github. [Fake Bills](https://medium.com/@dawndarasms/principal-component-analysis-pca-identifying-counterfeit-bills-a71ba615b65d)

##Kuhn, M. (2023). "C5.0 Decision Trees and Rule-Based Models" CranProject.org 

##Husejinovic, A.(2020) "Credit card fraud detection using naive Bayesian and C4.5 decision tree classifiers" Periodicals of Engineering & Natural Sciences.

##Gandi,R.(2018). "Support Vector Machine. Introduction to Machine Learning Algorithms" [SVM from Scratch](https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47)



```{r installing libraries}

library(dplyr) # for data manipulation
library(stringr) # for data manipulation
library(caret) # for sampling
library(caTools) # for train/test split
library(ggplot2) # for data visualization
library(corrplot) # for correlations
library(Rtsne) # for tsne plotting
library(rpart)# for decision tree model
library(Rborist)# for random forest model
library(xgboost) # for xgboost model
library(C50)
library(caret)
library(ggplot2)
library(ROSE)
library(repr)
library(RColorBrewer)
library(e1071)
library(C50)
library(DMwR)
library(factoextra)
library(ROCR)
library(pROC)
library(nnet)
library(tidyr)

```

```{r bringing in data}

df <- read.csv("/cloud/project/creditcard.csv", header=TRUE, stringsAsFactors=FALSE)
head(df,5)

```

#EDAs - Exploratory Data Analysis

##Because the variables have already been normalized, the names are labeled as V1, V2, V3, etc and they are expressed on the same scale, which is why the means are all "0." This means that when we run our correlation, they will be largely uncorrelated. The only variables not scaled are Amount and Time, which we will do later.

```{r data types}

str(df)

```


```{r data summary/EDAs}

summary(df)

```

```{r looking for any missing}

sum(is.na(df))

```


##Now we knew because our data was normalized that there would be no missing, however the code above, if the data was not cleaned already would have told us if there were missing.

##When we run our summary we see that our variable for Amount has at least one, if not more, outliers, which could impact our analysis.

```{r frequency table for Class}

df$Class<-as.factor(df$Class)
prop.table(table(df$Class))

```

```{r barplot of Class}

counts <- table(df$Class)
barplot(counts, main="Fraud Distribution",
        xlab="Counts of Fraud", col="hot pink")

```

## From this we can see the very uneven representation of credit card fraud in the data, which is usual. This means that the prediction will always inclined towards the majority class. 

##We will have to address this. This imbalance in the dataset will lead to imbalance in u might see a high accuracy number, but we should not befooled by that number. Because the sensitivity is low. Therefore, we have to address this imbalance in the dataset before going further.


```{r distribution of Amount}

hist(df$Amount, col = 'pink', border = "white")

```

```{r Distribution of Time}

hist(df$Time, col = 'purple', border = "white")

```


```{r histograms of other continuous vars}


a <- colnames(df)[2:29] 
par(mfrow = c(1, 4))    
for (i in 1:length(a)){
  sub = df[a[i]][,1]   
  hist(sub, main = paste("Hist. of", a[i], sep = " "), xlab = a[i])
}

```

#Preprocessing/Cleaning the Data

##Now we are going to scale the variables Time and Amount, as the other variables are. We are using a Z Score formula to set the means to "0" and standardize the variables.

```{r scaling Amount and Time}

# applying scale function 
df[1 : 30] <- as.data.frame(scale(df[1 : 30])) 

# displaying result 
head(df,5)

```

#3 Splitting Data into Train/Test Samples

##Why are we splitting our data?  And why, in our next step are we adjusting the training data so that the "Class" variable, which represents Fraud, will be even in the training set, but will not be altered in the testing set?  One answer:  "Bayesian Statistics and machine learning."

##In Fisher Statistics, we set up a null hypothesis to be disproved according to a p-value. Fisher is based upon lab conditions where variables are controlled and manipulated.

##However,the goal of Bayesian ML is to estimate the likelihood is something that can be estimated from the training data. In machine learning, the model is essentially learning the parameters of the target from the training data.  The idea behind ML is that models are built upon the parameters of real-world estimations.

##When training a regular machine learning model, it's an iterative process which updates the model’s parameters in an attempt to maximize the probability of seeing the training data having already seen the model parameters.

```{r Splitting into Test/Train}

set.seed(1234)
index<-createDataPartition(df$Class,p=0.8,list=FALSE)
train<-df[index,]
test<-df [-index,]

# Get the count and proportion of each classes
prop.table(table(train$Class))
```
```{r count of train}

table(train$Class)
```


#4) Sampling/Under-Sampling/SMOTE

##There are a few ways to balance the dataset on the Class variable.  


##Upsampling - this method increases the size of the minority class by sampling with replacement so that the classes will have the same size. An advantage of using this method is that it leads to no information loss. The disadvantage of using this method is that, since oversampling simply adds replicated observations in original data set, it ends up adding multiple observations of several types, thus it often leads to overfitting.

##Downsampling - in contrast to the above method, this one decreases the size of the majority class to be the same or closer to the minority class size by just taking out a random sample. The problem with this method is that it decreases the size of the overall sample in the training set, which can be a problem sfor some algorithms.

##Hybrid Methods - These include ROSE and SMOTE. ROSE (Random oversampling examples), and SMOTE (Synthetic minority oversampling technique), they downsample the majority class, and create new artificial points in the minority class.

##We will be using downsampling.  Generally, downsampling is recommended over oversampling, as downsampling gives a more accurate representation of the cases within the dataset you are trying to detect/train for. 


```{r Distribution of Class before Undersampling}


ggplot(train,aes(x=Amount,y=Time,group=Class))+
  geom_point(aes(color=Class))+
  scale_color_brewer(palette="Accent")


```

##Now we are going to under-sample to balance our training set

```{r Downsampling to balance our data}
set.seed(111)
traindown<-downSample(x=train[,-ncol(train)],
                      y=train$Class)

table(traindown$Class)

```

##Now we will check the distribution again

```{r distribution of Class after undersampling}

ggplot(traindown,aes(x=Amount,y=Time,group=Class))+
  geom_point(aes(color=Class))+
  scale_color_brewer(palette="Accent")

```


##As we can see, in the undersampled, it appears more randomly distributed.

##5) Correlation Matrices 

##We are going to look at the correlation of the variables with regard particularly to fraud in the train sample which has been balanced (under sampled). Otherwise the sample will be impacted by the imbalance in the classes.

##Evaluating a Correlation Matrix ***In reading a correlation matrix < +/-.29 is considered a LOW correlation

##between +/- .30 and .49 is a MEDIUM correlation and

##between +/- .50 and 1.00 is a HIGH correlation

##Pearsons Correlation Test #Correlation Matrix for All Quantitative Variables #drop variables that are not quantitative or are categorical #We are also saving the correlation matrix itself as a dataframe for a visualization

```{r}
str(traindown)
```



```{r to run the correlation we have to change the factor Class to numeric}

#convert column 'Class' from factor to numeric
traindown$Class <- as.numeric(as.character(traindown$Class))

```



```{r checking that it went well}

class(traindown$Class)
```



```{r correlation}

cormat <- round(cor(traindown),
                digits = 2 # rounded to 2 decimals
)

print(cormat)
```




```{r}
M <-cor(traindown)
```


```{r correlation plot}

options(repr.plot.width=35, repr.plot.height=35)
corrplot(M, type="upper", order = "hclust", col=brewer.pal(n=8, name="PiYG"))
```

##V17, V14, V12 and V10 are negatively correlated. Notice how the lower these values are, the more likely the end result will be a fraud transaction.

##V2, V4, V11, and V19 are positively correlated. Notice how the higher these values are, the more likely the end result will be a fraud transactions

##Leaving Class as numeric for Clustering


```{r}
# K-means clustering
# +++++++++++++++++++++
km.res <- kmeans(traindown, 3, nstart = 10)

```

```{r}
# Visualize kmeans clustering
# use repel = TRUE to avoid overplotting
fviz_cluster(km.res, traindown[, -5], ellipse.type = "norm")

```
## Although the subsample is pretty small, the Kmeans algorithm is able to detect clusters pretty accurately

##This gives us an indication that further predictive models will perform pretty well in separating fraud cases from non-fraud cases.

##We will be changing Class variable back to factor because the algorithms we wish to use need to have a target variable that is a factor variable (Logistic, Decision Trees)


```{r change Class back to factor}

traindown$Class <- as.factor(traindown$Class)

```

##In many instances of modeling, we would want to detect the outliers and remove those cases, such as in psychological studies.  However, with several types of verticals, such as sensor fault detection, fraud detection, and disaster risk warning systems it's the outliers or anomalies that are of most interest, as they often indicate the unusual situation we are trying to detect. Again, this will also inform the algorithm choice.  However, we will run an outlier detection routine to view the outliers in our dataset.

```{r}

# create detect outlier function
detect_outlier <- function(x) {
  
  # calculate first quantile
  Quantile1 <- quantile(x, probs=.25)
  
  # calculate third quantile
  Quantile3 <- quantile(x, probs=.75)
  
  # calculate interquartile range
  IQR = Quantile3 - Quantile1
  
  # return true or false
  x > Quantile3 + (IQR * 1.5) | x < Quantile1 - (IQR * 1.5)
}

# create remove outlier function
remove_outlier <- function(traindown, columns = names(dataframe)) {
  
  # for loop to traverse in columns vector
  for (col in columns) {
    
    # remove observation if it satisfies outlier function
    traindown <- traindown[!detect_outlier(traindown[[col]]), ]
  }
  
  # return dataframe
  print("Remove outliers")
  head(traindown,5)
}

remove_outlier(traindown,c('V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26','V27','V28','Time','Amount'))


```


##6) Logistic Regression 

##We are going to fit a Logistic Regression Model.How do we know which algorithms to try? 

##When selecting an algorithm or family of algorithms, you consider a few things:

##The question that you are answering - in this case we are answering if a transaction is fraud or not fraud - this is a classification problem - classification problems solve discrete (yes/no or binary, or multiple classification outcome problems)

##We also must consider the type of data which we have - in our case we have continuous (not mixed) predictor variables and no missing.  Often times you might have missing data.  Sometimes you will choose to delete or impute those values.  Often times in fraud those missing values can be a signal.  That is for another discussion. Some algorithms handle missing data better than other. However, we do not have any. So, we do not have to consider this.

##Logistic regression is conceptually similar to linear regression, where linear regression estimates the target variable. Instead of predicting values, as in the linear regression, logistic regression would estimate the odds of a certain event occurring. 

##Since we are predicting fraud, with logistic regression we are actually estimating the odds of fraud occuring. 

```{r Selecting vars that were correlated for new training set}

subtraindown <-traindown %>% select(Class, V17, V14, V12, V10,V2, V4, V11, V19)

```


```{r Logistic regression - including highly correlated vars}

# Training model
LR_Model <- glm(formula = Class ~ V17 + V14 + V12 + V10 + V2 + V4 + V11 + V19, data = subtraindown,family = "binomial")


```
```{r}
### Validate on trainData

Valid_trainData <- predict(LR_Model, newdata = subtraindown, type = "response") #prediction threshold
Valid_trainData <- ifelse(Valid_trainData > 0.5, 1, 0)  # set binary 

#produce confusion matrix
confusion_Mat<- confusionMatrix(as.factor(subtraindown$Class),as.factor(Valid_trainData))


```



```{r}
print(confusion_Mat)
```
##Overall, this is a very high-performing model.  In looking at the confusion matrix where there is room for improvment is that there is some misclassification of fraud as non-fraud predictions.


```{r}
### Validate on validData
testData_Class_predicted <- predict(LR_Model, newdata = test, type = "response")  

testData_Class_predicted  <- ifelse(testData_Class_predicted  > 0.5, 1, 0)  # set binary prediction threshold

conMat<- confusionMatrix(as.factor(test$Class),as.factor(testData_Class_predicted))

Regression_Acc_Test <-round(conMat$overall["Accuracy"]*100,2)
paste('Model Test Accuracy =', Regression_Acc_Test) 

```


```{r produce prediction on test data}

# create roc curve
roc_object <- roc( test$Class, testData_Class_predicted)


# calculate area under curve
auc(roc_object)

```
##The ROC curve helps us visualize the trade-off between sensitivity (True Positive Rate) and specificity (1 - False Positive Rate) for various threshold values. A perfect classifier would have an ROC curve that passes through the top-left corner of the plot (100% sensitivity and 100% specificity). The area under the ROC curve (AUC) is a scalar value that summarizes the performance of the classifier. An AUC of 1.0 indicates a perfect classifier, while an AUC of 0.5 suggests that the classifier is no better than random chance.  Ours is 93%.

##The other "good news" is that because our prediction on the train and the prediction on the test are so close, the likelihood of over-fitting is less likely.


##SVM - or a Support Vector Machine is a Machine Learning algorithm.  The objective of the support vector machine algorithm is to find a hyperplane in an N-dimensional space(N — the number of features) that distinctly classifies the data points. To separate the two classes of data points, there are many possible hyperplanes that could be chosen. Our objective is to find a plane that has the maximum margin, i.e the maximum distance between data points of both classes. Maximizing the margin distance provides some reinforcement so that future data points can be classified with more confidence.



```{r SVM}

# fit the model using default parameters
SVM_model<- svm(Class ~ V17 + V14 + V12 + V10 + V2 + V4 + V11 + V19, data=subtraindown, kernel = 'radial', type="C-classification")


Valid_trainData <- predict(SVM_model, subtraindown) 

#produce confusion matrix
confusion_Mat<- confusionMatrix(as.factor(subtraindown$Class), as.factor(Valid_trainData))

print(confusion_Mat)

# output accuracy
AVM_Acc_Train <- round(confusion_Mat$overall["Accuracy"]*100,4)
paste('Model Train Accuracy =', AVM_Acc_Train)


```



##Again we have a highly accurate model with very few misclassifications, and those that are misclassified are fraud classified as non-fraud.



```{r predict on Test Data}

### Test on Test Data
Test_Fraud_predicted <- predict(SVM_model, test) #produce confusion matrix 

conMat<- confusionMatrix(as.factor(test$Class), as.factor(Test_Fraud_predicted))

# output accuracy
AVM_Acc_Test <- round(conMat$overall["Accuracy"]*100,4)
paste('Model Test Accuracy =', AVM_Acc_Test) 

# prediction accuracy on test 
SVM_Acc <- c(AVM_Acc_Train, AVM_Acc_Test)


```
```{r Both Train and Test Accuracy for SVM}

SVM_Acc

```






















