# Pandas works with data frames i.e spread sheets. We use python over excel bse it is 
#faster with huge chunks of data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style #make the grapgh look fancy
style.use('ggplot') 

web_stats = {'Day': [1,2,3,4,5,6],
             'Visitors':[34,65,36,78,23,56],
             'Bounce_rate': [24,53,25,75,16,66]
            }

#convert webstats to a dataframe
df = pd.DataFrame(web_stats)

print(df)    #prints data, visualisation with indexes

print(df.head()) #first five rows

print(df.tail()) #last five rows

print(df.tail(2)) # last 2 rows

# setting index, this returns a new data frame with the new index
print(df.set_index('Day'))
df.set_index('Day', inplace = True) #modifies the dataframe

#indexing a dataframe
df['Visitors']
df.Visitors   #dictionaries cannot do this
df[['Visitors','Bounce_rate']] #to reference moe than one column, put them in a list.
 
#converting to lists
df.Visitors.to_list()

#converting to arrays
#python natively has no arrays, we use numpy to convert data to arrays
print(np.array(df[['Visitors','Bounce_rate']]))

#pandas input/output. read official documentation on pandas I
df.to_csv('newcsv2.csv') #to csv file
df = pd.read_csv('newcsv2.csv') # back to df with indexes
df = pd.read_csv('newcsv2.csv',index_col = 0) # specify index

#to html
df.to_html('example.html')
print(df)

#label column
df.columns = ['column name']

#remove colums in csv file
df = pd.read_csv('newcsv2.csv', header = False)

#read columns without header
df = pd.read_csv('newcsv2.csv',names=['Date', 'HPI'], index_col = 0)

