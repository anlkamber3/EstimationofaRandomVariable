import numpy as np
import math

np.random.seed(0) #If you erase this line,the script produces new uniform random variables at the beginning of each run.

#Part 1
def uniform_sample_generator():
    return np.random.uniform(0, 1, 2)

z_0 = [] #Samples of two independent standard normal distributions

for i in range(1000):
    z_0.append(math.sqrt((-2) * math.log(uniform_sample_generator()[0])) * math.cos(2*math.pi * uniform_sample_generator()[1]))
z_0.sort() #For vector subtraction,it is needed.

#Part 2

#We have samples of standard normal distribution. We can use sample mean and sample variance to find mean and variance of X.

sample_Mean = sum(z_0)/len(z_0)

buffer = 0

for j in z_0:
    buffer += (j-sample_Mean)**(2)

sample_Variance = buffer/len(z_0)

# I've wanted to make X and Y highly correlated so I have picked rho = 0.9 such that. I have picked variance of Y as 1.3
variance_Of_Y = 1.3
mean_Of_Y = 0.01
cov_X_Y = 0.9*math.sqrt(sample_Variance)*math.sqrt(variance_Of_Y)
covariance_Matrix = [[sample_Variance,cov_X_Y],[cov_X_Y,variance_Of_Y]]

expected_Value_Vector = [sample_Mean,mean_Of_Y]

def bivariate_gaussian_random_variable():
    return np.random.multivariate_normal(expected_Value_Vector,covariance_Matrix)
#Creating samples for bivariate gaussian random variable

b_0 = []

for j in range(1000):
    b_0.append(bivariate_gaussian_random_variable())
# The best estimation for a random variable without any a posteriori information is mean of random variables.
#Sample Variance is simply the MSE of blind estimation since our estimation is sample mean.
print("Blind Estimation of X:",sample_Mean,", MSE:",sample_Variance)

#For X > mean(Y)/2
def func(samples,mean_of_Y):
    n=0
    summation = 0
    data = []
    for i in samples:
        if (i > (mean_of_Y/2)):
            summation+= i
            n+= 1
        else:
            continue
    data.append(summation/n) #Finding the expected value given A
    mse_summation = 0
    for j in samples:
        if (j>(mean_of_Y/2)):
            mse_summation += (j-(summation/n))**(2)
        else:
            continue
    data.append(mse_summation/n)
    return data

print("Assuming that X > mean(Y)/2, the estimation is:",func(z_0,mean_Of_Y)[0],", MSE:",func(z_0,mean_Of_Y)[1])

#Optimal estimation given Y suggests that we need to take expected value of X given Y for the minimum MSE. Since X,Y bivariate the expected value of
#X given Y is formulized.

def minimum_MSE_Estimate(value_Of_Given_Y):
    return sample_Mean + (0.9*((math.sqrt(sample_Variance))/math.sqrt(variance_Of_Y))*(value_Of_Given_Y-mean_Of_Y))

array_Of_Estimates = []
for i in b_0:
    array_Of_Estimates.append(minimum_MSE_Estimate(i[1]))
array_Of_Estimates.sort()

def func_2(array): # Finding  MSE
    summation= 0
    for i in range(len(z_0)):
        summation+=(z_0[i]-array[i])**(2)
    return summation/len(z_0)
print("MSE of Optimal Estimation Given Y:",func_2(array_Of_Estimates))

#Linear Estimation of X given Y

#Coefficients:
a = cov_X_Y/ variance_Of_Y
b = sample_Mean - a*mean_Of_Y
optimal_Linear_Estimates = []
for i in b_0:
    optimal_Linear_Estimates.append(a*i[1]+b)
optimal_Linear_Estimates.sort()
print("MSE of Linear Estimation of X given Y:",func_2(optimal_Linear_Estimates))

#MAP Estimate
#Since Joint PDF of X,Y is bivariate gaussian, the PDF of X given Y is a Gaussian.
def mean_of_PDF_of_X_given_Y(value_of_Given_Y):
    return sample_Mean + ((0.9)*((math.sqrt(sample_Variance))/math.sqrt(variance_Of_Y))*(value_of_Given_Y-mean_Of_Y))
#Since it is Gaussian, maximum of PDF is occurs at mean.

array_Of_Estimates_MAP = []

for j in b_0:
    array_Of_Estimates_MAP.append(mean_of_PDF_of_X_given_Y(j[1]))
array_Of_Estimates_MAP.sort()

print("MSE of MAP Estimation of X given Y:",func_2(array_Of_Estimates_MAP))

#ML Estimate
#In ML we have no apriori information about X. However we know Y, and they are bivariate gaussian. So to maximize PDF of Y given X,
#we need to take its derivative w.r.t x and equalize to 0.

def maximizing_x(value_of_Given_Y):
    return ((1/(0.9))*((math.sqrt(sample_Variance))/math.sqrt(variance_Of_Y))*(value_of_Given_Y-mean_Of_Y))

array_Of_Estimates_ML = []

for z in b_0:
    array_Of_Estimates_ML.append(maximizing_x(z[1]))
array_Of_Estimates_ML.sort()
print("MSE of ML Estimation of X given Y:",func_2(array_Of_Estimates_ML))
