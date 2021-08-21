#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import math
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from gspread_dataframe import get_as_dataframe, set_with_dataframe
from functools import reduce
import re
import matplotlib.pyplot as plt
import statsmodels.api as sm
import Levenshtein

#put this as shared email on docs of interest: testadvocacy@states-update-237403.iam.gserviceaccount.com
#go here for more information: https://www.twilio.com/blog/2017/02/an-easy-way-to-read-and-write-to-a-google-spreadsheet-in-python.html

# use creds to create a client to interact with the Google Drive API
scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('50 States Update-a7933c0a355c.json', scope)
client = gspread.authorize(creds)

# Find a workbook by name and open the first sheet
# Make sure to use test-271@luminous-shadow-238018.iam.gserviceaccount.com

#open output.csv
pd.set_option('display.float_format', lambda x: '%.f' % x)
sheet_2 = client.open("Output (2)").sheet1

# Extract and print all of the values
list_of_hashes = sheet_2.get_all_values()
output = pd.DataFrame(list_of_hashes)
output.columns = output.iloc[0]
output.drop(0, axis = 0, inplace = True)
output.drop(columns =[''], inplace = True)

for x in ['NCES School ID', 'Total Students All Grades (Excludes AE)', 
          'Grades 9-12 Students','Latitude', 'Longitude']:
    output[x] = pd.to_numeric(output[x], errors = 'coerce')

sheet_3 = client.open("ELSI_csv_export_6369881421855947098470").sheet1
list_of_hashes = sheet_3.get_all_values()
ELSI = pd.DataFrame(list_of_hashes)
ELSI.columns = ELSI.iloc[0]
ELSI.drop(0, axis = 0, inplace = True)

print(ELSI)
#take out processed incompletes
output = output[output['Highest Standard'] != 'Incomplete Information']


#find ratio of 9-12 to total students at each high school
def high_school(df):
    try:
        ratio = float(df['Grades 9-12 Students [Public School] 2016-17'])/float(df['Total Students All Grades (Excludes AE) [Public School] 2016-17'])
        return ratio
    except: 
        return np.nan
    
ELSI['9-12 ratio'] = ELSI[['Grades 9-12 Students [Public School] 2016-17', 'Total Students All Grades (Excludes AE) [Public School] 2016-17']].apply(high_school, axis = 1)


#remove all incomplete schools that aren't 100% 9-12
ELSI_copy = ELSI.copy(deep = True)
ELSI = ELSI[ELSI['9-12 ratio'] >= .5]

#clean NCES ID columns to transform into floats
def replace_cols(item):
    try:
        if item.startswith('="'):
            return item[2:-1]
    except AttributeError:
        return item
    return item

cols = ELSI.columns
ELSI[cols] = ELSI[cols].applymap(replace_cols)
ELSI['School ID - NCES Assigned [Public School] Latest available year'] = pd.to_numeric(ELSI['School ID - NCES Assigned [Public School] Latest available year'], errors = 'coerce')

#drop and append ELSI data
NCES_list = output['NCES School ID'].tolist()
NCES_list = pd.unique(NCES_list).tolist()
ELSI.set_index('School ID - NCES Assigned [Public School] Latest available year', inplace = True)
ELSI.drop(NCES_list, errors ='ignore', inplace = True)
ELSI.reset_index(inplace = True)

def clean(item):
    temp = item.find('[Public School]')
    if temp != -1:
        return item[:temp-1]
    else:
        return item
    
ELSI['Final Standard'] = 'Incomplete Information'
ELSI.reset_index(inplace = True)
ELSI.rename(columns = clean, inplace = True)
output.rename(columns = clean, inplace = True)
ELSI.rename(columns = {'School Name': 'School Name_x', 'School ID - NCES Assigned': 'NCES School ID', 'Web Site URL': 'Link to School Website'}, inplace = True)
output = output.append(ELSI)
sheet = client.open("testadvocacymerge1").sheet1

# Extract and print all of the values
list_of_hashes = sheet.get_all_values()
advocacy = pd.DataFrame(list_of_hashes)
advocacy.columns = advocacy.iloc[0]
advocacy.drop(0, axis = 0, inplace = True)
advocacy.drop_duplicates(subset = ['School ID - NCES Assigned [Public School] Latest available year'], 
                                  keep = 'last', inplace = True)


public = advocacy[advocacy['School Status'] == 'Public']
public.drop(columns = ['Address', 'City', 'State', 'Zip Code'], inplace = True)
private = advocacy[advocacy['School Status'] == 'Private']

#take advocacy sheet and merge with output
public['School ID - NCES Assigned [Public School] Latest available year'] = pd.to_numeric(public['School ID - NCES Assigned [Public School] Latest available year'], errors = 'coerce')
public['Latitude (testadvocacy)'] = pd.to_numeric(public['Latitude (testadvocacy)'], errors = 'coerce')
output = output.merge(public, left_on = 'NCES School ID', right_on = 'School ID - NCES Assigned [Public School] Latest available year', how = 'left') 
#change 'Standard' column to Incomplete Information
def fill(item):
    try:
        if math.isnan(item):
            return "Incomplete Information"            
    except TypeError:
        return item
    return item
output['Standard'] = output['Standard'].apply(fill)

#change standards columns to numbers
def standard_class(item):
    if item == 'Gold Standard':
        return 3
    elif item == 'Silver Standard':
        return 2
    elif item == 'Bronze Standard':
        return 1
    else:
        return 0
output['Standard'] = output['Standard'].apply(standard_class) 
output['Highest Standard']= output['Highest Standard'].apply(standard_class)

#compare two columns
def standard_compare(df):
    if df['Standard'] != 0:
        return df['Standard']
    else:
        return df['Highest Standard']
    
output['Final Standard'] = output[['School Name_x_x', 'Highest Standard', 'Standard']].apply(standard_compare, axis = 1)

#convert back 
def standard_class_2(item):
    if item == 3:
        return "Gold Standard"
    elif item == 2:
        return 'Silver Standard'
    elif item == 1:
        return "Bronze Standard"
    else:
        return "Incomplete Information"
    
output['Final Standard'] = output['Final Standard'].apply(standard_class_2)
output.sort_values(by = ['School Name_x_x', 'Final Standard'], inplace = True)


def course_compare(df):
    try:
        if math.isnan(df['Class Name_x']):
            return df['Proposed Course Name']
    except(TypeError):
        return df['Class Name_x']
    return df['Class Name_x']
    
output['Final Class Name'] = output[['Class Name_x', 'Proposed Course Name']].apply(course_compare, axis = 1)
output.sort_values(by = ['School Name_x_x', 'Final Class Name'], inplace = True)

def catalog_compare(df):
    try:
        if math.isnan(df['Link to Course Catalog_x']):
            #print(df['School Name_x_x'], df['Link to Course Catalog_x'], df['Link to Course Catalog (testadvocacy)'])
            return df['Link to Course Catalog (testadvocacy)']
    except(TypeError):
        return df['Link to Course Catalog_x']
    return df['Link to Course Catalog_x']
    
output['Link to Course Catalog_x'] = output[['School Name_x_x', 'Link to Course Catalog_x', 'Link to Course Catalog (testadvocacy)']].apply(catalog_compare, axis = 1)
output.sort_values(by = ['School Name_x_x', 'Link to Course Catalog_x'], inplace = True)


def latitude_compare(df):
    try:
        if math.isnan(df['Latitude']):
            return df['Latitude (testadvocacy)']
        else:
            return df['Latitude']
    except(TypeError):
        return df['Latitude']

output['Latitude (testadvocacy)'] = pd.to_numeric(output['Latitude (testadvocacy)'], errors = 'coerce')
#print(output[['School Name_x_x', 'Latitude (testadvocacy)', 'Latitude']].dropna())
output['Latitude'] = output[['School Name_x_x', 'Latitude', 'Latitude (testadvocacy)']].apply(latitude_compare, axis = 1)
output.sort_values(by = ['School Name_x_x', 'Latitude'], inplace = True)

def longitude_compare(df):
    try:   
        if math.isnan(df['Longitude']):
            #print(df['School Name_x_x'], df['Longitude'], df['Longitude (testadvocacy)'])
            return df['Longitude (testadvocacy)']
        else:
            return df['Longitude']
    except(TypeError):
        return df['Longitude']

output['Longitude (testadvocacy)'] = pd.to_numeric(output['Longitude (testadvocacy)'], errors = 'coerce')
output['Longitude'] = output[['Longitude', 'Longitude (testadvocacy)']].apply(longitude_compare, axis = 1)
output.sort_values(by = ['School Name_x_x', 'Longitude'], inplace = True)

output.replace(', nan', '', inplace = True)
output['Longitude'] = pd.to_numeric(output['Longitude'], errors = 'coerce')
output['Latitude'] = pd.to_numeric(output['Latitude'], errors = 'coerce')

#merge other stats forgotten in original iteration
output.drop(columns = ['Free and Reduced Lunch Students', 'Urban-centric Locale'], inplace = True)
ELSI_copy.rename(columns = {'School ID - NCES Assigned [Public School] Latest available year': 'NCES School ID', 'Free and Reduced Lunch Students [Public School] 2016-17': 'Free and Reduced Lunch Students',
                            'Total Students All Grades (Excludes AE) [Public School] 2016-17': 'Total Students K-12',
                           'Urban-centric Locale [Public School] 2016-17': 'Urban-centric Locale', 'Grade 12 Students [Public School] 2016-17': 'Grade 12 Students'}, inplace = True)
ELSI_keep = ['NCES School ID', 'Free and Reduced Lunch Students', 'Total Students K-12','Urban-centric Locale', 'Grade 12 Students']
ELSI_copy = ELSI_copy[ELSI_keep]
ELSI_copy['Total Students K-12'] = pd.to_numeric(ELSI_copy['Total Students K-12'], errors = 'coerce')
ELSI_copy['Free and Reduced Lunch Students'] = pd.to_numeric(ELSI_copy['Free and Reduced Lunch Students'], errors = 'coerce')
ELSI_copy['Free and Reduced Lunch Ratio'] = ELSI_copy['Free and Reduced Lunch Students']/ELSI_copy['Total Students K-12']
cols = ELSI_copy.columns
ELSI_copy[cols] = ELSI_copy[cols].applymap(replace_cols)
ELSI_copy['NCES School ID'] = pd.to_numeric(ELSI_copy['NCES School ID'], errors = 'coerce')

output = output.merge(ELSI_copy, left_on = 'NCES School ID', right_on = 'NCES School ID', how = 'left')

#take remaining private sheet, create into dataframe that matches output, and then append
private.rename(columns = ({'School Name_x': 'School Name_x_x', 'Standard': 'Final Standard', 
                           'School ID - NCES Assigned [Public School] Latest available year':'NCES School ID', 
                           'Address': 'Location Address 1', 'City': 'Location City', 'State': 'Location State Abbr', 
                           'Zip Code': 'Location ZIP', 'Longitude (testadvocacy)': 'Longitude', 
                           'Latitude (testadvocacy)' : 'Latitude' , 'Proposed Course Name': 'Class Name_x'}), inplace = True)

#NOTE : append private later


#ELSI read in and clean
keep = ['NCES School ID','School Name_x_x','Description of course_x', 'Final Standard', 
      'Full-Time Equivalent (FTE) Teachers', 'Grades 9-12 Students', 'Location Address 1', 
      'Location City', 'Location State Abbr', 'Location ZIP', 'County Name', 'Latitude', 'Longitude', 
      'Year of Course Catalog', 'Link to School Website', 'Link to Course Catalog_x', 
      'Free and Reduced Lunch Students', 'Urban-centric Locale', 'Final Class Name',
      'Total Students K-12', 'Free and Reduced Lunch Ratio', 'Grade 12 Students_x']

output = output[keep]
output.rename(columns = {'Final Class Name': 'Class Name_x'}, inplace = True)
output['School Name_x_x'] = output['School Name_x_x'].apply(lambda x: x.title() if isinstance(x, str) else x)
output['Location City'] = output['Location City'].apply(lambda x: x.title() if isinstance(x, str) else x)
output.sort_values(by = ['School Name_x_x'], kind = 'quicksort', inplace = True)
output.drop_duplicates(subset = ['NCES School ID', 'School Name_x_x','Class Name_x'], keep = 'first', inplace = True)
output['Free and Reduced Lunch Students'] = pd.to_numeric(output['Free and Reduced Lunch Students'], errors='coerce')
path2='/Users/sidsharma/Downloads/output full 2018.csv'
output.to_csv(path2)
print('First part done')


# In[ ]:





# In[ ]:





# In[7]:


#aggregate course names, and use that one for map 

pd.options.display.float_format = '{:,.2f}'.format

def standard_to_number(item):
    if item == 'Gold Standard':
        return 1
    else:
        return 0


def testaddcomma(series):
    return reduce(lambda x, y: str(x) + ', ' + str(y), series)

def findnotna(series):
    if len(series) > 1:
        for x in series:
            if math.isnan(x) == False:
                return x
    return series.iloc[0]

state_abbrev = {
        'AK': 'Alaska',
        'AL': 'Alabama',
        'AR': 'Arkansas',
        'AS': 'American Samoa',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'GU': 'Guam',
        'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        'MP': 'Northern Mariana Islands',
        'MS': 'Mississippi',
        'MT': 'Montana',
        'NA': 'National',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        'PR': 'Puerto Rico',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VI': 'Virgin Islands',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'
}

aggregate = output.groupby(['NCES School ID','School Name_x_x']).agg({'Final Standard': lambda x: x.iloc[0],
                                                                      'Class Name_x': testaddcomma, 
                                                                      'Grades 9-12 Students': lambda x: x.iloc[0],
                                                                      'Full-Time Equivalent (FTE) Teachers': lambda x: x.iloc[0],
                                                                      'Location Address 1': lambda x: x.iloc[0],
                                                                      'Location City': lambda x: x.iloc[0],
                                                                     'Location State Abbr': lambda x: x.iloc[0],
                                                                     'Location ZIP': lambda x: x.iloc[0],
                                                                     'County Name': lambda x: x.iloc[0],
                                                                     'Latitude': findnotna,
                                                                     'Longitude': findnotna,
                                                                     'Link to School Website': lambda x: x.iloc[0],
                                                                     'Link to Course Catalog_x': lambda x: x.iloc[0], 
                                                                     'Urban-centric Locale': lambda x: x.iloc[0],
                                                                     'Free and Reduced Lunch Students': lambda x: x.iloc[0],
                                                                     'Total Students K-12':lambda x: x.iloc[0],
                                                                     'Free and Reduced Lunch Ratio': lambda x: x.iloc[0],
                                                                     'Grade 12 Students_x': lambda x: x.iloc[0] })

for x in ['Grades 9-12 Students', 'Full-Time Equivalent (FTE) Teachers', 'Location ZIP', 
       'Latitude', 'Longitude', 'Total Students K-12',
       'Free and Reduced Lunch Ratio', 'Grade 12 Students_x']:
    aggregate[x] = pd.to_numeric(aggregate[x], errors = 'coerce')

aggregate['School Status'] = 'Public'
aggregate.reset_index(inplace = True)
private['School Name_x_x'] = private['School Name_x_x'] + ' (Independent)'
aggregate = aggregate.append(private)
aggregate.set_index(['NCES School ID', 'School Name_x_x'], inplace = True)

aggregate['Location State Abbr'] = aggregate['Location State Abbr'].apply(lambda x: x.strip() if isinstance(x, str) else x)
aggregate.dropna(subset = ['Latitude'], inplace = True)

#combine FAFSA filings
FAFSA = pd.read_excel('HS_ARCHIVE05312019.xls', skiprows = 3)

FAFSA.rename(columns = {'Applications\nSubmitted\nMay31  2019': 'Applications Submitted May 31, 2019'},
            inplace = True)
keep = ['Name', 'City', 'State', 'Applications Submitted May 31, 2019']
FAFSA = FAFSA[keep]
#take out schools with 0 to 5 applications
FAFSA = FAFSA[FAFSA['Applications Submitted May 31, 2019'] != '<5']
for x in ['Name', 'City']:
    FAFSA[x] = FAFSA[x].astype('str')
    FAFSA[x] = FAFSA[x].apply(lambda x: x.lower())
    FAFSA[x] = FAFSA[x].apply(lambda x: x.strip())
    
aggregate_copy1 = aggregate.reset_index()
for x in ['School Name_x_x', 'Location City']:
    aggregate_copy1[x] = aggregate_copy1[x].astype('str')
    aggregate_copy1[x] = aggregate_copy1[x].apply(lambda x: x.lower())
    aggregate_copy1[x] = aggregate_copy1[x].apply(lambda x: x.strip())

FAFSA['State'] = FAFSA['State'].apply(lambda x: x.strip())

school_suffix = [ "unified school district",
                                "school district",
                                "public schools",
                                "sr. h.s.",
                                "sr.h.s",
                                "sr.hs",
                                "sr h.s.",
                                "sr. hs",
                                "sr hs",
                                "s.r./j.r.",
                                "sr/jr",
                                "s.r/j.r",
                                "s.r.",
                                "j.r.",
                                "sr",
                                "jr",
                                "H.S.",
                                "h.s.",
                                "HS",
                                "high school",
                                "magnet school",
                                 "magnet",
                                "high",
                                "school",
                                " sch"]

def replace_suffix(item):
    for x in school_suffix:
        item = item.replace(x, '')
    return item

aggregate_copy1['School Name_clean'] = aggregate_copy1['School Name_x_x'].apply(replace_suffix)
aggregate_copy1['School Name_clean'] = aggregate_copy1['School Name_clean'].apply(lambda x: x.strip())
FAFSA['Name_clean'] = FAFSA['Name'].apply(replace_suffix)
FAFSA['Name_clean'] = FAFSA['Name_clean'].apply(lambda x: x.strip())



def FAFSA_match(df):
    temp_df = aggregate_copy1[aggregate_copy1['Location City'] == df['City']]
    temp_df2 = temp_df[temp_df['Location State Abbr'] == df['State']]
    NCES_id = 'No Match'
    lscore_temp = 4
    if len(temp_df2) > 0:
        for index, row in temp_df2.iterrows():
            lscore = Levenshtein.distance(row['School Name_clean'], df['Name_clean'])
            if lscore < 4:
                if lscore < lscore_temp:
                    NCES_id = row['NCES School ID']
                    if lscore == 0:
                        return NCES_id
                    lscore_temp = lscore
    return NCES_id

FAFSA['NCES Match'] = FAFSA[['Name_clean', 'City', 'State']].apply(FAFSA_match, axis = 1)
FAFSA = FAFSA[FAFSA['NCES Match'] != 'No Match']
FAFSA['NCES Match'] = pd.to_numeric(FAFSA['NCES Match'], errors = 'coerce')
FAFSA = FAFSA[['NCES Match', 'Applications Submitted May 31, 2019']]

#merge with aggregate

aggregate_copy1['NCES School ID'] = pd.to_numeric(aggregate_copy1['NCES School ID'], errors = 'coerce')
aggregate_copy1 = aggregate_copy1.merge(FAFSA, left_on = 'NCES School ID', right_on = 'NCES Match', how = 'left')
aggregate_copy1.to_csv('Aggregate with FAFSA.csv')
    
    

free_lunch = aggregate[aggregate['Final Standard'] != 'Incomplete Information']
aggregate_incomplete_removed = free_lunch.copy(deep = True)
free_lunch['Free and Reduced Lunch Ratio'] = free_lunch['Free and Reduced Lunch Ratio'].astype('float64')
free_lunch.dropna(subset = ['Free and Reduced Lunch Ratio', 'Final Standard'], inplace = True)
required_states = {'AL': 'N', 'WY': 'N', 'TN': 'N', 'UT': 'N', 'MO': 'N', 'VA': 'N', '': 'N'}
free_lunch.replace({'Location State Abbr': required_states}, inplace = True)
free_lunch = free_lunch[free_lunch['Location State Abbr'] != 'N']

def standard_to_number(item):
    if item == 'Gold Standard':
        return 1
    else:
        return 0


#regression
aggregate_incomplete_removed.dropna(subset = ['Free and Reduced Lunch Ratio'], inplace = True)
X = aggregate_incomplete_removed['Free and Reduced Lunch Ratio']
Y = aggregate_incomplete_removed['Final Standard'].apply(standard_to_number)

X = sm.add_constant(X)
 
model = sm.OLS(Y.astype(float), X.astype(float)).fit()
predictions = model.predict(X) 
 
print_model = model.summary()
print(print_model)

plt.scatter(aggregate_incomplete_removed['Free and Reduced Lunch Ratio'], aggregate_incomplete_removed['Final Standard'])
plt.xlabel('Free and Reduced Lunch Ratio', fontsize=14)
plt.ylabel('Standard 1 for Gold, 0 for not', fontsize=14)
plt.show()



aggregate_gold = aggregate[aggregate['Final Standard'] == 'Gold Standard']
aggregate_gold.reset_index(inplace = True)

def distance(df, lat2 = 35.216817, lon2 = -80.838271): 
    radius = 6371 # km
    lat1 = df['Latitude']
    lon1 = df['Longitude']
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1))         * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c
    return d

def closest_school(df):
    closest = 100000000
    closest2 = 100000000
    closest3 = 100000000
    dis = 0
    school = ''
    lat1 = df['Latitude']
    lon1 = df['Longitude']
    for x in range(0, len(aggregate_gold) - 1):
        lat2 = aggregate_gold['Latitude'].iloc[x]
        lon2 = aggregate_gold['Longitude'].iloc[x]
        diff_lat = float(lat2) - float(lat1)
        diff_lon = float(lon2) - float(lon1)
        if (abs(diff_lat)> 2) or (abs(diff_lon) >2):
            continue
        dis = distance(float(lat1), float(lon1), float(lat2), float(lon2))
        if (dis < closest) and (dis != 0):
            closest = dis
            df['Closest Gold Standard'] = aggregate_gold['School Name_x_x'].iloc[x]
            df['Closest Gold Standard Location'] = aggregate_gold['Location City'].iloc[x] + ', ' + aggregate_gold['Location State Abbr'].iloc[x]
            df['Closest Distance'] = closest
    return df


aggregate.drop(columns = ['Link to Course Catalog (testadvocacy)'], inplace = True)

path3='/Users/sidsharma/Downloads/Aggregate Full List.csv'
aggregate.to_csv(path3)
print('Full List saved')

path4='/Users/sidsharma/Downloads/Tableau Map Aggregate.csv'

tableau_map = aggregate.reset_index()
sheet_1 = client.open("Test Aggregate Push").sheet1
set_with_dataframe(sheet_1, tableau_map)
aggregate.to_csv(path4)


aggregate['Grades 9-12 Students'] = aggregate['Grades 9-12 Students'].astype('float64')
aggregate_copy = aggregate.copy(deep = True)



incomplete_removed = aggregate[aggregate['Final Standard'] != 'Incomplete Information']

'''
state_of_choice = 'IA'
state_filtered = incomplete_removed[incomplete_removed['Location State Abbr'] == state_of_choice]
path4 = '/Users/sidsharma/Downloads/state_filtered(' + state_of_choice + ').csv'
state_filtered.to_csv(path4)
print('State filtered list downloaded')

#specific_pd = aggregate_copy[aggregate_copy['Location State Abbr'] == state_of_choice]
specific_pd = aggregate_copy
specific_pd['Distance from Des Moines FinCamp'] = specific_pd[['Latitude', 'Longitude']].apply(distance, axis = 1)
specific_pd['Distance from Des Moines FinCamp'] = specific_pd['Distance from Des Moines FinCamp']/1.609
specific_pd = specific_pd[specific_pd['Distance from Des Moines FinCamp'] <= 50]
print(specific_pd['Distance from Des Moines FinCamp'])
path5 = '/Users/sidsharma/Downloads/Des Moines FinCamp Prospecting List.csv'
specific_pd.to_csv(path5)
print('Prospecting list downloaded')

'''
df1 = pd.DataFrame()

incomplete_removed = incomplete_removed[incomplete_removed['School Status'] == 'Public']
for i in state_abbrev:
    state_temp = incomplete_removed[incomplete_removed['Location State Abbr'] == i]
    gold_temp = state_temp[state_temp['Final Standard'] == 'Gold Standard']
    silver_temp = state_temp[state_temp['Final Standard'] == 'Silver Standard']
    bronze_temp = state_temp[state_temp['Final Standard'] == 'Bronze Standard']
    if len(state_temp) != 0:
        df2 = pd.DataFrame(({'State': [state_abbrev[i]], 'Covered Students': state_temp['Grades 9-12 Students'].sum(),
                            'Gold Standard Students': [gold_temp['Grades 9-12 Students'].sum()],
                            'Silver Standard Students': [silver_temp['Grades 9-12 Students'].sum()],
                            'Bronze Standard Students': [bronze_temp['Grades 9-12 Students'].sum()],
                            'Gold Standard Schools': [len(gold_temp)],
                            'Silver Standard Schools': [len(silver_temp)],
                            'Bronze Standard Schools': [len(bronze_temp)]}))
        df1 = df1.append(df2)
df1.reset_index(inplace = True)
df1.drop(columns = 'index', inplace = True)

summary_stats = client.open("50 States Summary Statistics (2018-2019)").sheet1
set_with_dataframe(summary_stats, df1)

path5 = '/Users/sidsharma/Downloads/Summary Stats updated.csv'
df1.to_csv(path5)
print("Summary Stats updated")


'''path4='/Users/sidsharma/Downloads/people-1732955-589.csv'
persons = pd.read_csv(path4)
#persons['Organization - NCES ID'] = persons['Organization - NCES ID'].apply(lambda x: x if x.isalnum() else 'No NCES ID')

def findsplit(item):
    try:
        if isinstance(item, str):
            if item.find(',') != -1:
                list = item.split(',')
                return float(list[0])
            item = re.sub("[^0-9]", "", item)
            return float(item[:12])
    except(ValueError):
        if isinstance(item, float) or isinstance(item, int):
            return item
        else:
            return -1

def emailsplit(item):
    if isinstance(item, str):
        if item.find(',') != -1:
            list = item.split(',')
            return list[0]
        else:
            return item
    else: 
        return 'No Email'
    
persons['Organization - NCES ID'] = persons['Organization - NCES ID'].apply(findsplit)
persons['Organization - NCES ID'] = pd.to_numeric(persons['Organization - NCES ID'], errors = 'coerce')
persons['Person - Email'] = persons['Person - Email'].apply(emailsplit)
#persons = persons[persons['Person - Email'] != 'No Email']
aggregate.replace({'Location State Abbr': state_abbrev}, inplace = True)
persons = persons.merge(aggregate, left_on = 'Organization - NCES ID', right_on = 'NCES School ID', how = 'inner')
keep = ['Person - Name', 'Person - Organization', 'Person - Email', 'Organization - NCES ID', 'Grades 9-12 Students', 'Location Address 1', 'Location City', 'Location State Abbr',
       'Location ZIP', 'County Name', 'Link to School Website']
persons = persons[keep]
'''
#read in aggregate for states
state = pd.read_csv('50 States Summary Statistics (2018-2019) - Sheet1.csv')
state.rename(columns = {'Unnamed: 0': 'State Name'}, inplace = True)
us_rate = (state['Total Gold Standard Students (9-12)'].iloc[50])/(state['Total Checked Students (9-12)'].iloc[50])

def divide_two(df):
    x = df['Total Gold Standard Students (9-12)']
    y = df['Total Checked Students (9-12)']
    return x/y


state['Gold Standard percent'] = state[['Total Gold Standard Students (9-12)', 'Total Checked Students (9-12)']].apply(divide_two, axis = 1)
state.reset_index(inplace = True)


def standard_message(df):
    if (us_rate < df['Gold Standard percent']):
        return 'In other news, ' + df['State Name'] + " performed phenomenally in this year's assessment of personal finance quality with a total of " + str(int(df['Total Gold Standard Students (9-12)'])) + ' Gold Standard students over ' + str(int(df['Gold Standard 2018-19'])) + ' Gold Standard schools, ' + str(int(df['Silver Standard 2018-19'])) + ' Silver Standard schools, and ' + str(int(df['Bronze 2018-19'])) + " Bronze Standard schools. It is teachers like you that have helped your school educate so many schools in personal finance. Congratulations!"
    else:
        return (df['State Name'] + ' finished the year with ' + str(int(df['Gold Standard 2018-19'])) + ' Gold Standard schools, '+ str(int(df['Silver Standard 2018-19']))
                 + ' Silver Standard schools, and ' + str(int(df['Bronze 2018-19'])) + ' Bronze Standard schools. In total, ' + str(int(df['Total Covered Students(9-12)'])) + ' were covered in your state.')

state = state.truncate(after = 50)

state['Standard Message'] = state[['Possible Rank', 'Total Covered Students(9-12)', 'State Name', 'Gold Standard percent', 'Total Gold Standard Students (9-12)', 'Gold Standard 2018-19', 'Silver Standard 2018-19', 'Bronze 2018-19']].apply(standard_message, axis =1)

email_list = persons.merge(state, left_on = 'Location State Abbr', right_on = 'State Name', how = 'left')
keep = ['Person - Name', 'Person - Email', 'Person - Organization',
       'Organization - NCES ID', 'Grades 9-12 Students',
       'Location City', 'Location State Abbr',
       'Link to School Website', 'Total Checked Students (9-12)', 'Total Students (9-12)',
       'Total Covered Students(9-12)', 'Total Gold Standard Students (9-12)',
       'Gold Standard Charters', 'Gold Standard 2018-19', 'Silver Standard 2018-19', 
       'Bronze 2018-19','Non-existent personal finance course or incomplete information 2018-19',
       'Point Average 2018-2019', 'Possible Rank', 'Gold Standard percent',
       'Standard Message']
email_list = email_list[keep]
path5 = '/Users/sidsharma/Downloads/50 States Email List.csv'
email_list.to_csv(path5)

email_list.drop_duplicates(subset = ['Organization - NCES ID', 'Person - Organization'], keep = 'first', inplace = True)
#print(email_list['Grades 9-12 Students'].sum()) 

#determine number of non-required gold standard schools
required_states = {'Alabama': 'N', 'Tennessee': 'N', 'Utah': 'N', 'Missouri': 'N', 'Virginia': 'N', '': 'N'}
aggregate.replace({'Location State Abbr': required_states}, inplace = True)
aggregate = aggregate[aggregate['Final Standard'] == 'Gold Standard']
aggregate = aggregate[aggregate['Location State Abbr'] != 'N']
print('Number of Gold Standard schools total:', len(aggregate))


# In[ ]:





# In[ ]:





# In[ ]:




