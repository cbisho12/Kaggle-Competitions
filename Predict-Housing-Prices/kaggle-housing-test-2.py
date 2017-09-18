
# Kaggle Competition to Predict Housing Prices
# =============
# 
# Data set Downloaded from https://www.kaggle.com/c/house-prices-advanced-regression-techniques
# ------------
# 

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# The features and their types to load the CSV file
type_dict = {'MSSubClass':np.dtype('S') , 
             'MSZoning':np.dtype('S'),
             'Street':np.dtype('S'),
             'Alley':np.dtype('S'),
             'LotShape':np.dtype('S'),
             'LandContour':np.dtype('S'),
        'Utilities':np.dtype('S'),
        'LotConfig':np.dtype('S'),
        'LandSlope':np.dtype('S'),
        'Neighborhood':np.dtype('S'),
        'Condition1':np.dtype('S'),
        'Condition2':np.dtype('S'),
        'BldgType':np.dtype('S'),
        'HouseStyle':np.dtype('S'),
        'RoofStyle':np.dtype('S'),
        'RoofMatl':np.dtype('S'),
        'Exterior1st':np.dtype('S'),
        'Exterior2nd':np.dtype('S'),
        'MasVnrType':np.dtype('S'),
        'ExterQual':np.dtype('S'),
        'ExterCond':np.dtype('S'),
        'Foundation':np.dtype('S'),
        'BsmtQual':np.dtype('S'),
        'BsmtCond':np.dtype('S'),
        'BsmtExposure':np.dtype('S'),
        'BsmtFinType1':np.dtype('S'),
        'BsmtFinType2':np.dtype('S'),
        'Heating':np.dtype('S'),
        'HeatingQC':np.dtype('S'),
        'CentralAir':np.dtype('S'),
        'Electrical':np.dtype('S'),
        'KitchenQual':np.dtype('S'),
        'Functional':np.dtype('S'),
        'FireplaceQu':np.dtype('S'),
        'GarageType':np.dtype('S'),
        'GarageFinish':np.dtype('S'),
        'GarageQual':np.dtype('S'),
        'GarageCond':np.dtype('S'),
        'PavedDrive':np.dtype('S'),
        'PoolQC':np.dtype('S'),
        'Fence':np.dtype('S'),
        'MiscFeature':np.dtype('S'),
        'MoSold':np.dtype('S'),
        'SaleType':np.dtype('S'),
        'SaleCondition':np.dtype('S'),
        'LotArea':np.float,
        'YearBuilt':np.float,
        'YearRemodAdd':np.float,
        'BsmtFinSF1':np.float,
        'BsmtFinSF2':np.float,
        'BsmtUnfSF':np.float,
        'TotalBsmtSF':np.float,
        '1stFlrSF':np.float,
        '2ndFlrSF':np.float,
        'LowQualFinSF':np.float,
        'GrLivArea':np.float,
        'BsmtFullBath':np.float,
        'BsmtHalfBath':np.float,
        'FullBath':np.float,
        'HalfBath':np.float,
        'Bedroom':np.float,
        'KitchenAbvGr':np.float,
        'TotRmsAbvGrd':np.float,
        'Fireplaces':np.float,
        'GarageCars':np.float,
        'GarageArea':np.float,
        'WoodDeckSF':np.float,
        'OpenPorchSF':np.float,
        'EnclosedPorch':np.float,
        '3SsnPorch':np.float,
        'ScreenPorch':np.float,
        'PoolArea':np.float,
        'MiscVal':np.float,
        'YrSold':np.int,
        'MasVnrArea':np.dtype('S'),
        'LotFrontage':np.dtype('S'),
        'GarageYrBlt':np.dtype('S')}


# These are the categorical data i.e. the non-numerical features that need to be one hot encoded
categ_dict = {'MSSubClass':np.dtype('S') , 
             'MSZoning':np.dtype('S'),
             'Street':np.dtype('S'),
             'Alley':np.dtype('S'),
             'LotShape':np.dtype('S'),
             'LandContour':np.dtype('S'),
        'LotConfig':np.dtype('S'),
        'LandSlope':np.dtype('S'),
        'Neighborhood':np.dtype('S'),
        'BldgType':np.dtype('S'),
#        'HouseStyle':np.dtype('S'),
#        'RoofStyle':np.dtype('S'),
#        'RoofMatl':np.dtype('S'),
        'CentralAir':np.dtype('S'),
        'GarageType':np.dtype('S'),
        'GarageFinish':np.dtype('S'),
        'PavedDrive':np.dtype('S'),
        'Fence':np.dtype('S'),
        'MoSold':np.dtype('S'),
        'SaleType':np.dtype('S'),
        'SaleCondition':np.dtype('S'),
         'YrSold':np.dtype('S') }


# These are the numerical features
num_dict = {
        '1stFlrSF':np.float,
        '2ndFlrSF':np.float,
        'LowQualFinSF':np.float,
        'GrLivArea':np.float,
        'FullBath':np.float,
        'HalfBath':np.float,
        'BedroomAbvGr':np.float,
        'KitchenAbvGr':np.float,
        'TotRmsAbvGrd':np.float,
        'Fireplaces':np.float,
        'WoodDeckSF':np.float,
        'OpenPorchSF':np.float,
        'EnclosedPorch':np.float,
        '3SsnPorch':np.float,
        'ScreenPorch':np.float,
        'PoolArea':np.float,
        'MiscVal':np.float,
        'OverallQual':np.float,
        'OverallCond':np.float,
        'AgeWhenSold':np.float,
        'AgeRemodAdd':np.float}


# Read in the training set and the test set ffrom csv files
filename_train = 'train.csv'
full_train_data = pd.read_csv(filename_train,dtype=type_dict ) 
filename_test = 'test.csv'
full_test_data = pd.read_csv(filename_test,dtype=type_dict ) 

mm_train=len(full_train_data['LotArea'])
mm_test=len(full_test_data['LotArea'])
full_train_data['AgeWhenSold'] = full_train_data['YrSold']-full_train_data['YearBuilt']
full_test_data['AgeWhenSold'] = full_test_data['YrSold']-full_test_data['YearBuilt']
full_train_data['AgeRemodAdd'] = full_train_data['YearRemodAdd']-full_train_data['YearBuilt']
full_test_data['AgeRemodAdd'] = full_test_data['YearRemodAdd']-full_test_data['YearBuilt']


# Add the numerical categories to New DataFrame
indx_dict={}
#del flat_data1
flat_train_data=pd.DataFrame(data=0.0,index=xrange(0,mm_train),columns=['LotArea'] )
flat_train_data['LotArea']=full_train_data['LotArea']
indx_dict['LotArea'] = 0
for x in num_dict:
    flat_train_data[x] = full_train_data[x]
i_count=np.int(1)
for x in num_dict:
    indx_dict[x]=i_count
    i_count+=1

# features to leave out
leave_dict={'nan':0,
          '150':1,
          'TenC':2,
           '2.5Fin':3,
           'Typ':4,
           'Normal':5}

# One hot encode and add the categorical features to the training set
i_count=len(indx_dict)
for x in categ_dict:
    s=full_train_data[x][:]
    c=s.unique()
    for i in range(0,len(c)):
        # This IF removes unwanted features
        if str(c[i]) not in leave_dict: 
            v = str(x) + '_' + str(c[i])
            indx_dict[v]=i_count
            i_count=i_count+1
            flat_data2=pd.DataFrame(data=0.0,index=xrange(0,mm_train),columns=[v] )
            flat_data2
            for j in range(0,mm_train):
                if full_train_data[x][j] == c[i]:
                    flat_data2[v][j] = 1.0
                else:
                    flat_data2[v][j] = 0.0
            flat_train_data[v]=flat_data2
            del flat_data2
np.shape(flat_train_data)


# for Condition1 and Condition2
cond_dict = {
       'Artery':0, 'Feedr':1,
       'RRNn':2, 'RRAn':3,
       'PosN':4, 'PosA':5,
       'RRNe':6, 'RRAe':7 }
i_count=len(indx_dict)
for x in cond_dict:
    v = 'Condition_' + str(x)
    indx_dict[v]=i_count
    i_count = i_count + 1
    flat_data2=pd.DataFrame(data=0.0,index=xrange(0,mm_train),columns=[v] )
    for i in range(mm_train):
        if (full_train_data['Condition1'][i] == x):
            flat_data2[v][i] += 1.0
        if (full_train_data['Condition2'][i] == x):
            flat_data2[v][i] += 1.0
    flat_train_data[v]=flat_data2
    del flat_data2


def polynomial_df(df_single, x_dict, feature, degree):
    # assume that degree >= 1
    # and set poly_sframe['power_1'] equal to the passed feature
    # first check if degree > 1
    mm = len(x_dict)
    if degree > 1:
        # then loop over the remaining degrees:
        for power in range(2, degree+1): 
            # first we'll give the column a name:
            name = feature+'_p' + str(power)
            x_dict[name]= mm + power-2
            #print (name, x_dict[name])
            # then assign poly_sframe[name] to the appropriate power of feature
            df_single[name] = df_single[feature].apply(lambda x: x**power)
    return 


#degr_dict = {}
degr_dict = {'OverallQual':0,
              'OverallCond':1,
             'BedroomAbvGr':2,
             'TotRmsAbvGrd':3,
             'GrLivArea':4,
              'ExterQual':5,
              'ExterCond':6 }

# turn listed quality feature into numerical feature
Qual_dict = { 'Ex':4,
             'Gd':3,
             'TA':2,
               'Fa':1,
               'Po':0,
		'nan':1 } # when NA assume slightly negetive 
Qual_categ_dict = {'ExterQual':0,
              'ExterCond':1,
               'HeatingQC':2,
                'BsmtQual':3,
                'BsmtCond':4,
                'GarageCond':5}
i_count=len(indx_dict)
for x in Qual_categ_dict:
    indx_dict[x]=i_count
    i_count = i_count + 1
    flat_data2=pd.DataFrame(data=0.0,index=xrange(0,mm_train),columns=[x] )
    for i in range(mm_train):
    	flat_data2[x][i]=Qual_dict[str(full_train_data[x][i])]
    flat_train_data[x]=flat_data2
    del flat_data2

# Make Functional a numerical feature
funct_dict = {  'Typ':7,
           'Min1': 6,
           'Min2':5,
           'Mod':4,
           'Maj1':3,
           'Maj2':2,
           'Sev':1,
           'Sal':0,
            'nan': 5} # when NA assume slightly negetive 
i_count=len(indx_dict)
indx_dict['Functional']=i_count
flat_data2=pd.DataFrame(data=0.0,index=xrange(0,mm_train),columns=['Functional'] )
for i in range(mm_train):
	x =np.str(full_train_data['Functional'][i])
	flat_data2['Functional'][i]=funct_dict[x]
flat_train_data['Functional']=flat_data2
del flat_data2


for x in degr_dict:
    polynomial_df(flat_train_data,indx_dict,x,5)


# Now create the test_data np array. First flatten and one hot encode the test data set.
# Copy over numerical(continuous) features
flat_test_data=pd.DataFrame(data=0.0,index=xrange(0,mm_test),columns=['LotArea'] )
flat_test_data['LotArea']=full_test_data['LotArea']

for x in num_dict:
    flat_test_data[x] = full_test_data[x]


# one hot encode categorical data
for x in categ_dict:
    s=full_test_data[x][:]
    c=s.unique()
    for i in range(0,len(c)):
        if str(c[i]) not in leave_dict: 
            v = str(x) + '_' + str(c[i])
            #print(indx_dict[v])
            flat_data2=pd.DataFrame(data=0.0,index=xrange(0,mm_test),columns=[v] )
            flat_data2
            for j in range(0,mm_test):
                if full_test_data[x][j] == c[i]:
                    flat_data2[v][j] = 1.0
                else:
                    flat_data2[v][j] = 0.0
            flat_test_data[v]=flat_data2
            del flat_data2
np.shape(flat_test_data)


# for Condition1 and Condition2
i_count=len(indx_dict)
for x in cond_dict:
    v = 'Condition_' + str(x)
    #indx_dict[v]=i_count
    #i_count = i_count + 1
    flat_data2=pd.DataFrame(data=0.0,index=xrange(0,mm_test),columns=[v] )
    for i in range(mm_test):
        if (full_test_data['Condition1'][i] == x):
            flat_data2[v][i] += 1.0
        if (full_test_data['Condition2'][i] == x):
            flat_data2[v][i] += 1.0
        #if (flat_data2[v][i] > 1.0):
            #print (v,i,flat_data2[v][i])
    flat_test_data[v]=flat_data2


for x in Qual_categ_dict:
    flat_data2=pd.DataFrame(data=0.0,index=xrange(0,mm_train),columns=[x] )
    for i in range(mm_test):
    	flat_data2[x][i]=Qual_dict[str(full_test_data[x][i])]
    flat_test_data[x]=flat_data2
    del flat_data2


# Make Functional a numerical feature in test data
flat_data2=pd.DataFrame(data=0.0,index=xrange(0,mm_test),columns=['Functional'] )
for i in range(mm_test):
	x=np.str(full_test_data['Functional'][i])
	flat_data2['Functional'][i]=funct_dict[x]
flat_test_data['Functional']=flat_data2
del flat_data2

indx_dict_c = {}
for x in degr_dict:
    polynomial_df(flat_test_data,indx_dict_c,x,5)


# initialize the training and testing numpy arrays
a_train=len(full_train_data['LotArea'])
a_test=len(full_test_data['LotArea'])
b=len(indx_dict)
train_data = np.zeros((a_train,b))
test_data = np.zeros((a_test,b))


for x in indx_dict:
    m=indx_dict[x]
    train_data[:,m]=flat_train_data[x]
    test_data[:,m]=flat_test_data[x]


train_labels = np.zeros(a_train)
train_labels[:] = full_train_data['SalePrice'][:]


# Create linear regression object
#regr = linear_model.LinearRegression()
regr = linear_model.Ridge (alpha = 10.0,tol=0.0001) #, max_iter=5000)
# Train the model using the training sets
regr.fit(train_data, train_labels)

y_predict = regr.predict(test_data)


for x in range(len(y_predict)):
    print(y_predict[x] )





