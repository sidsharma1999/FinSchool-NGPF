#!/usr/bin/env python
# coding: utf-8

# In[109]:


#import statements
import pandas as pd
import Levenshtein
import re
import numpy as np
from functools import reduce

#get files from computer
path='/Users/sidsharma/Downloads/Mass1819.csv'
path2 ='/Users/sidsharma/Downloads/Cleaned - Master 50 State Data - Massachusetts.csv'

#read csv and fill in missing data
Mass1819 = pd.read_csv(path)
Mass1819['Updated in'] = "2018"
Mass1819.fillna(method = 'ffill', limit = 1, inplace = True)
Mass1718 = pd.read_csv(path2)
Mass1718['Updated in'] = "2017"


#combine, sort, and make everything lowercase
df = Mass1819.append(Mass1718, sort = False)
df.sort_values('School Name', inplace = True)
df.drop(columns = ['Requirement', 'Notes'], inplace = True)

#change out states
states = {
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

df['State'].replace(states, inplace = True)

#combine, sort, and make everything lowercase
state = df.loc[1]['State'].iloc[0]
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
df['Class Name'] = df['Class Name'].apply(lambda x: x.lower() if isinstance(x,str) else x)
temp = ''
#print(len(df))
df2 = pd.DataFrame()

#check if a string has either a digit or a roman numeral
def contains(item):
    if re.search('[0-9]', item):
        return True
    elif item.find('ii') != -1:
        return True
    return False

#iterate through dataframe
for i in df['School Name']:
    #check if we already iterated through this one
    if temp == i:
        continue
        
    #if not already done, divide dataframe for specific school into 2018 and 2017
    else:
        temp = i
        tempdf18 = df[(df['School Name'] == i) & (df['Updated in'] == "2018")]
        tempdf17 = df[(df['School Name'] == i) & (df['Updated in'] == "2017")]   
        tempdf18.reset_index(inplace = True)
        tempdf17.reset_index(inplace = True)
        
        #if there is at least 2 schools in each dataframe, iterate through
        if (len(tempdf18) > 1) & (len(tempdf17) > 1):
            for z in range(0, len(tempdf18)):
                class18name = tempdf18.loc[z]['Class Name']
                class18descript = tempdf18.loc[z]['Description of course']
                for w in range(0, len(tempdf17)):
                    class17name = tempdf17.loc[w]['Class Name']
                    class17descript = tempdf17.loc[w]['Description of course']
                    flag = False
                    
                    #run Levenshtein score on name and description of class
                    try:
                        if ((contains(class18name) == False) and (contains(class17name) == False)):
                            if (Levenshtein.distance(class18name,class17name)  <= 5 and 
                                Levenshtein.distance(class18name,class17name)  > 0):
                                #print(class18name, class17name, temp)
                                tempdf17.replace({class17name:class18name}, inplace = True)
                                flag = True
                        if (Levenshtein.distance(class18descript,class17descript)  <= 30 and 
                            Levenshtein.distance(class18descript,class17descript)  > 0 and 
                            Levenshtein.distance(class18name,class17name) != 0 and
                            flag == False):
                                #print("description-based", class18name, class17name)
                                tempdf18.replace({class18name:class17name}, inplace = True)
                    except(TypeError):
                        break
                        
        #concatenate cleaned and combined data into one dataframe                        
        df2 = pd.concat([df2, tempdf18, tempdf17])

df2.reset_index(inplace = True)


#keep good track for further summation
df2['Class Name Count'] = 1

#aggregation functions to combine various series of data
def testaddsemicolon(series):
    return reduce(lambda x, y: str(x) + '; ' + str(y), series)

def testaddcomma(series):
    return reduce(lambda x, y: str(x) + ', ' + str(y), series)

def lenseries(series):
    return len(series)

#recapitalize
df2['Class Name'] = df2['Class Name'].apply(lambda x: x.title() if isinstance(x, str) else x)
tempsheet = df2.copy(deep = True)
tempsheet = tempsheet[tempsheet['Updated in'] != '2017']

#determine standard
def standard(df):
    try: 
        temp = float(df['Duration'])
    except ValueError:
        if ((df['Grad'] == 'E' or df['Grad'] == 'Cluster') or (df['Grad'] == 'R' or df['Grad'] == 'Required')):
            return "At least Bronze Standard"
        return 'Incomplete Information'
    
    if (temp>=.5 and (df['Grad'] == 'R' or df['Grad'] == 'Required') and (df['Type'] == 'S' or df['Type'] == 'Standalone')):
         return "Gold Standard"
    elif (temp>=.5 and (df['Grad'] == 'E' or df['Grad'] == 'Cluster') and (df['Type'] == 'S' or df['Type'] == 'Standalone')):
        return "Silver Standard"
    elif ((df['Grad'] == 'E' or df['Grad'] == 'Cluster') or (df['Grad'] == 'R' or df['Grad'] == 'Required')):
        return "Bronze Standard"
    else: 
        return "Incomplete Information"
    
tempsheet['Standard'] = tempsheet[['Duration', 'Grad', 'Type', 'Class Name']].apply(standard, axis = 1)

#group by and aggregate
df2 = df2.groupby(['NCES School ID', 'School Name', 'Class Name']).agg(({'Class Name Count': lenseries, 'Description of course': testaddsemicolon, 
                                                                  'Year of Course Catalog': testaddcomma, 'Updated in': testaddcomma}))
df2 = df2[df2['Updated in'] != "2017"]
df2.rename({'Class Name Count': 'If 2, was offered last year'}, axis = 'index', inplace = True)

keep = ['NCES School ID','Street.Address', 'Students.', 'School Name', 
       'Link to School Website', 'Link to Course Catalog',
       'Source', 'Class Name', 'Description of course', 'Duration', 'Grad',
       'Type', 'Year of Course Catalog', 'Class Name Count', 'Standard']
tempsheet = tempsheet[keep]
tempsheet = tempsheet.groupby(['NCES School ID', 'School Name']).agg(({'Class Name': testaddcomma, 'Class Name Count': np.sum,
                                                           'Link to Course Catalog': lambda x: x.iloc[0], 
                                                           'Description of course': testaddsemicolon,
                                                           'Grad': testaddcomma, 'Type': testaddcomma,
                                                           'Standard': testaddcomma}))

def highest_level(item):
    if isinstance(item, str):                                                            
        if item.find('S') != -1:
            return 'Standalone'
        elif item.find('E') != -1:
            return 'Umbrella'
    return 'No Access'                                                             

def highest_standard(item):
    if isinstance(item, str):
        if item.find('Gold Standard') != -1:
            return 'Gold Standard'
        elif item.find('Silver Standard') != -1:
            return 'Silver Standard'
        elif item.find('At least Bronze Standard') != -1:
            return 'At least Bronze Standard'
        elif item.find('Bronze Standard') != -1:
            return 'Bronze Standard'
    return 'Incomplete Information'
                                

tempsheet['Highest Level'] = tempsheet['Type'].apply(highest_level)
tempsheet['Highest Standard'] = tempsheet['Standard'].apply(highest_standard)


#process NCES file
ELSI = pd.read_csv('ELSI_csv_export_6368705505283686917463.csv', skiprows = 6)

#clean up column names
def clean(item):
    temp = item.find('[Public School]')
    if temp != -1:
        return item[:temp-1]
    else:
        return item

#clean up file
ELSI.rename(columns = clean, inplace = True)
ELSI.drop(ELSI.columns[4], axis=1, inplace = True)
print(ELSI.columns)
ELSI['State'] = ELSI['State Name'].apply(lambda x: x.title() if isinstance(x, str) else x)
ELSI = ELSI[ELSI['State Name'] == state]
tempsheet = tempsheet.merge(ELSI, left_on = 'NCES School ID', right_on = 'hey', how = 'left')
  

#analysis
required = len(tempsheet[tempsheet['Grad'] == 'R'])
cluster = len(tempsheet[tempsheet['Grad'] == 'E'])
followup = len(tempsheet[tempsheet['Grad'] == 'F'])
na = len(tempsheet[tempsheet['Grad'] == 'N']) + len(tempsheet[tempsheet['Grad'] == 'NA'])
standalone = len(tempsheet[tempsheet['Type'] == 'S'])
umbrella = len(tempsheet[tempsheet['Type'] == 'E'])
unique = len(df2[df2['Class Name Count'] == 1])/(len(df2))
total = len(tempsheet)


try:
    path3='/Users/sidsharma/Downloads/Master Summary .csv'
    summary = pd.read_csv(path3)
except(FileNotFoundError):
    d = ({'State': [state], 'Percent unique to this year' : [unique], 'Required courses' : [required]
      , 'Cluster Courses': [cluster], 'Follow-Up Required': [followup],
      'Relevant courses not found': [na], 'Standalone': [standalone], 
      'Umbrella': [umbrella], 'Total number of courses': [total]})
    summary = pd.DataFrame(data = d)
    summary.to_csv(path3)

d = ({'State': [state], 'Percent unique to this year' : [unique], 'Required courses' : [required]
      , 'Cluster Courses': [cluster], 'Follow-Up Required': [followup],
      'Relevant courses not found': [na], 'Standalone': [standalone], 
      'Umbrella': [umbrella], 'Total number of courses': [total]})
add = pd.DataFrame(data = d)
summary = summary.append(add, sort = 'False')
summary.drop_duplicates(subset = 'State', keep = 'last', inplace = True)
summary.to_csv(path3)
print(d)

#send back to computer
path = '/Users/sidsharma/Downloads/CourseMasterList/'
path += state + ' - Course Master List.csv'
df2.to_csv(path)
            


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




