from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from scipy.stats import mode
import pandas as pd
import pickle

dataSize = 2985217
properties_df = pd.read_csv('properties_2016.csv')

categoryNames = list(properties_df)
custom = ['regionidzip', 
          'parcelid',
          'poolcnt',
          'fireplacecnt']
binary = ['fireplaceflag',
          'hashottuborspa',
          'pooltypeid10',
          'pooltypeid2'
          'pooltypeid7',
          'taxdelinquencyflag']

toDeleteCategories = ['propertycountylandusecode',
                      'taxdelinquencyyear',
                      'propertyzoningdesc']
categoricalFeatures = ['airconditioningtypeid',
                       'architecturalstyletypeid',
                       'buildingclasstypeid',
                       'decktypeid',
                       'heatingorsystemtypeid',
                       'propertylandusetypeid',
                       'storytypeid',
                       'typeconstructiontypeid']
nonNumerical = categoricalFeatures + toDeleteCategories + custom + binary
numericalFeatures = [key for key in categoryNames if key not in nonNumerical]

propertiesGroupedByZip = properties_df.groupby('regionidzip')

categByCategoryZip = pickle.load(open( "categByCategoryZip.p", "rb" ))

numerByCategoryZip = {}
for numer in numericalFeatures:
    numerByCategoryZip[numer] = propertiesGroupedByZip[numer].mean()

mostCommonZip = int(properties_df['regionidzip'].mode())
properties_df['regionidzip'] = properties_df['regionidzip'].fillna(mostCommonZip)

properties_df['taxdelinquencyflag'] = properties_df['taxdelinquencyflag'].fillna(0)
properties_df['taxdelinquencyflag'] = properties_df['taxdelinquencyflag'].replace("Y", 1)
print pd.Series.value_counts(properties_df['taxdelinquencyflag'])

properties_df['fireplaceflag'] = properties_df['fireplaceflag'].replace(True, 1)
properties_df['fireplaceflag'] = properties_df['fireplaceflag'].fillna(0)
print pd.Series.value_counts(properties_df['fireplaceflag'])

properties_df['hashottuborspa'] = properties_df['hashottuborspa'].replace(True, 1)
properties_df['hashottuborspa'] = properties_df['hashottuborspa'].fillna(0)
print pd.Series.value_counts(properties_df['hashottuborspa'])

for pooltype in ['pooltypeid10','pooltypeid2','pooltypeid7','poolcnt', 'fireplacecnt']:
    properties_df[pooltype] = properties_df[pooltype].fillna(0)

numericElemsAvgsGlobal = {}
categorElemsModesGlobal = {}
for numer in numericalFeatures:
    numericElemsAvgsGlobal[numer] = pd.Series.mean(properties_df[numer])
for categ in categoricalFeatures:
    categorElemsModesGlobal[categ] = pd.Series.mode(properties_df[categ])[0]

for index, row in properties_df.iterrows():
    zipCode = row['regionidzip']
    for numer in numericalFeatures:
        if np.isnan(row[numer]):
            if zipCode in numerByCategoryZip[numer]:
                row[numer] = numerByCategoryZip[numer][zipCode]
            else:
                row[numer] = numericElemsAvgsGlobal[numer]
    for categ in categoricalFeatures:
        if np.isnan(row[categ]):
            if zipCode in categByCategoryZip[categ]:
                row[categ] = categByCategoryZip[categ][zipCode]
            else:
                row[categ] = categByCategoryZip[categ]
pickle.dump(properties_df, open( "almostProcessedData.p", "wb" ) )
