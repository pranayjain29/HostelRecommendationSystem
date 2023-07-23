import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cufflinks as cf
import pickle
import math
from statistics import mean

room_data=pd.read_csv("C:/Users/prana/Downloads/2 ND FLOOR STATICS.csv",encoding= 'unicode_escape')
room_data.drop(['Unnamed: 5','Unnamed: 6','Unnamed: 7','Unnamed: 8','Unnamed: 4'],axis=1,inplace=True)
room_data.dropna(inplace=True)
rooms=room_data['ROOM '].unique()
room_data.drop('ROOM TYPE',axis=1,inplace=True)
alloted={1:[],2:[],3:[],4:[]}
for rno in rooms:
  room_list=[]
  temp_df=room_data[room_data['ROOM ']==rno]
  for j in temp_df.index:
    room_list.append(temp_df.loc[j]['REG  NO'])
  key=len(room_list)
  alloted[key].append(room_list)


df2=pd.DataFrame()

df22=pd.read_csv("C:/Users/prana/OneDrive/Desktop/Dataset_TARP.csv",encoding= 'unicode_escape')
df2c=df22.copy()

'''
tempere=[]
for i in room_data.index:
    regno=room_data.loc[i]['REG  NO']
    tempere.append(df22[df22['REGISTER NUMBER']==regno])
    
df22=pd.concat(tempere)
'''

# perform inner join on 'REGISTER NUMBER' and 'REG NO' columns
merged_df = pd.merge(df22, room_data, how='inner', left_on='REGISTER NUMBER', right_on='REG  NO')
# keep only columns from the original df dataframe
df22 = merged_df[df22.columns]

#FOR ROOM_DATA
merged_df = pd.merge(df22, room_data, how='inner', left_on='REGISTER NUMBER', right_on='REG  NO')
room_data = merged_df[room_data.columns]
room_data.drop_duplicates(inplace=True)

df2=pd.DataFrame()
#df22=pd.read_csv('/Dataset_TARP(Final).csv',encoding= 'unicode_escape')
temper=[]
for i in df22.index:
    for j in df22.columns:
        slotss=df22.loc[i,'Slots'].split('+')
        for k in slotss:
            temper.append(pd.DataFrame(data=[[df22.loc[i,'REGISTER NUMBER'],k]],columns=['REGISTER NUMBER','Slots']))
df2=pd.concat(temper)
df2=df2.drop_duplicates()
df2=df2.reset_index(drop=True)

df=pd.DataFrame()
temp=[]
nameslist = df2['REGISTER NUMBER'].unique()
count=0
for i in df2['REGISTER NUMBER'].unique():
    #print(df2[df2['Name']==i].reset_index(drop=True).T)
    tempdf=df2[df2['REGISTER NUMBER']==i].reset_index(drop=True).T
  #  tempdf = 
    temp.append(tempdf.drop('REGISTER NUMBER',axis=0))

df = pd.concat(temp)
df.index=nameslist

df=df.fillna(0)

df4=pd.DataFrame()

for i in df.index:
    for j in df.columns:
        df4.loc[i,df.loc[i,j]]=1
df4=df4.fillna(0)
df4['REGISTER NUMBER']=df4.index
df4.reset_index(drop=True,inplace=True)
df4.drop(0,axis=1,inplace=True)
df4.drop('NIL',axis=1)

def sigm(row):
    for i,item in enumerate(row):
        row[i]=1/(1+math.exp(-3*row[i]))
    return row

# Create an empty dataframe with no columns or rows
df = {'Monday':pd.DataFrame(),'Tuesday':pd.DataFrame(),'Wednesday':pd.DataFrame(),'Thursday':pd.DataFrame(),'Friday':pd.DataFrame()}


for j in df.keys():
    col=['REGISTER NUMBER']
    for i in range(8,20):
        col.append(str(i))
    df[j] = pd.DataFrame(columns=col, dtype=float)

a={'Monday':['A1','L1','F1','L2','D1','L3','TB1','L4','TG1','L5','S11','L6','A2','L31','F2','L32','D2','L33','TB2','L34','TG2','L35','S3','L36'],
   'Tuesday':['B1','L7','G1','L8','E1','L9','TC1','L10','TAA1','L11','-','L12','B2','L37','G2','L38','E2','L39','TC2','L40','TAA2','L41','S1','L42'],
   'Wednesday':['C1','L13','A1','L14','F1','L15','TD1','L16','TBB1','L17','-','L18','C2','L43','A2','L44','F2','L45','TD2','L46','TBB2','L47','S4','L48'],
   'Thursday':['D1','L19','B1','L20','G1','L21','TE1','L22','TCC1','L23','-','L24','D2','L49','B2','L50','G2','L51','TE2','L52','TCC2','L53','S2','L54'],
   'Friday':['E1','L25','C1','L26','TA1','L27','TF1','L28','TDD1','L29','S15','L30','E2','L55','C2','L56','TA2','L57','TF2','L58','TDD2','L59','-','L60']}



map={'Monday':{},'Tuesday':{},'Wednesday':{},'Thursday':{},'Friday':{}}
for i in a.keys():
    t=8
    for j in range(0,len(a[i]),2):
        map[i][a[i][j]]='{}'.format(t)
        map[i][a[i][j+1]]='{}'.format(t)
        t+=1

col=['8','9','10','11','12','13','14','15','16','17','18']
data_tt=[[1,1,1,0,0,0,0,0,0,0,0],[0,1,1,1,1,0,0,0,0,0,0],[1,1,1,0,0,0,0,0,1,1,0],[1,0,1,1,0,0,0,0,1,1,0],[1,1,1,1,0,0,1,1,0,0,0]]


data_names=['Pranayyy']
data_days=['Monday','Tuesday','Wednesday','Thursday','Friday']


ind=[['Pranayyy','Pranayyy','Pranayyy','Pranayyy','Pranayyy'],
     ['Monday','Tuesday','Wednesday','Thursday','Friday',]]
df=pd.DataFrame(data_tt,index=ind,columns=col)

ind2=[['Monday','Monday','Monday','Monday','Monday','Monday','Monday','Monday','Monday','Monday','Monday'],
      ['8','9','10','11','12','13','14','15','16','17','18']]

col2=['Pranayyy','Pranay2','Pranay3']
data_tt2=[[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0]]
dftry=pd.DataFrame(data_tt2,index=ind2,columns=col2)
dftry=dftry.T

for i in range(len(df4)):
    for k in data_days:
      for j in df4.columns:
        
        if(j=='REGISTER NUMBER'):
            continue
        
        if(df4.iloc[i][j]==0):
            continue
        
        try:
            #print(df4.loc[i,'Name'])
            df.loc[(df4.loc[i,'REGISTER NUMBER'],k),map[k][j]]=1
            dftry.loc[df4.loc[i,'REGISTER NUMBER'],(k,map[k][j])]=1
          
        except:
          #print(j)
          continue

df=df.fillna(0)
dftry=dftry.fillna(0)
df=df[5:]
dftry=dftry[3:]
dftry['REGISTER NUMBER']=dftry.index
dftry['REGISTER NUMBER']=dftry['REGISTER NUMBER'].astype(str)

#Recommendation Part
dftry = dftry.sample(frac = 1).reset_index(drop=True)
df1=dftry.copy()
df1=df1.T
df1=df1.reset_index(drop=True)
df1.columns=df1.iloc[len(df1)-1]
df1.drop(index=df1.index[len(df1)-1],axis=0,inplace=True)
df1=df1.reset_index(drop=True)

df1=df1.astype(int)
df2=df1.corr().apply(lambda row: sigm(row))


feature_df=dftry.copy()
avail=[0]*dftry.shape[0]
rooms=[0]*dftry.shape[0]
dftry[('Avail')]=avail
dftry[('Rooms')]=rooms
feature_df=feature_df.drop(['REGISTER NUMBER'],axis=1)

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import neighbors

scaler = MinMaxScaler()
feature_arr = scaler.fit_transform(feature_df)

model = neighbors.NearestNeighbors(n_neighbors=150 , algorithm = 'ball_tree')
model.fit(feature_arr)

pickle.dump(model, open('model.pkl','wb'))
pickle.dump(feature_df, open('feature.pkl','wb'))
pickle.dump(dftry, open('df.pkl','wb'))
pickle.dump(df2, open('df2.pkl','wb'))

dist_mat , id_list_mat = model.kneighbors(feature_arr)

def rec_mates(name):
    r_list = []
    match_list=[]
    r_idx = dftry[dftry['REGISTER NUMBER'] == name].index
    r_idx = r_idx[0] #iff multiple index found
    count=0
    for num in id_list_mat[r_idx]:
        if((dftry.iloc[num].Avail!=1)[0]):
         r_list.append(dftry.iloc[num]['REGISTER NUMBER'][0])
         match_list.append(df2.loc[name][dftry.iloc[num]['REGISTER NUMBER']][0])
         count=count+1
    return r_list,match_list

flag=1
details={};


#alloted={2:None,3:None,4:None}
nameslist=room_data['REG  NO'].unique()

rec_hour=[]
flag=1

for name in nameslist:
    
    if(dftry.loc[dftry[dftry['REGISTER NUMBER']==name].index,'Avail'].iloc[0]==1):
        continue
    
    alloted={2:None,3:None,4:None}
    rec_,match_ = rec_mates(name)
    print(rec_)
    print("next")
    sel_list=[]
    try:
        for i in range(0,3):
         #   print(str(rec_[i]) + "       " + str(match_[i]))
           if(i >= len(rec_)):
               break
           sel_list.append(str(rec_[i]))
           
    except ValueError:
        break
    room=4
    for i in sel_list:
        dftry.loc[dftry[dftry['REGISTER NUMBER']==i].index,'Avail']=1 
       # dftry.loc[dftry[dftry['Name']==i].index,'Rooms']=x
    

    #rec_.remove(name)
    #rec_ = [ele for ele in rec_ if ele not in dell]
    details[name]=sel_list
    
    sel_list.append(name)
    alloted[room]=sel_list
    dfinal=[]
    
    for i in range(4,1,-1):
        if alloted[i]!=None:
            dfinal.append(dftry[dftry['REGISTER NUMBER'].isin(alloted[i])].reset_index(drop=True))
        else:
            dfinal.append(pd.DataFrame())


    for k in range(len(dfinal)):
        if dfinal[k].empty:
            continue
        totalh=11*5
        for i in dfinal[k].columns:
            if i==('REGISTER NUMBER',''):
                continue
            valid=0
            sum=0
            for j in range(0,len(dfinal[k])):
                sum += dfinal[k].loc[j,i]
            valid=(int) (sum/len(dfinal[k]))
            if(valid==1):
                totalh=totalh-1    #2/5
        rec_hour.append(totalh)
    

'''
xaxis=[i for i in range(0,29)]
plt.plot(xaxis,rec_hour)
plt.ylim([15, 55])
plt.grid()
plt.show()
'''

dfinal=[]
alloted={1:[],2:[],3:[],4:[]}
rooms=room_data['ROOM '].unique()
for rno in rooms:
  room_list=[]
  temp_df=room_data[room_data['ROOM ']==rno]
  for j in temp_df.index:
    room_list.append(temp_df.loc[j]['REG  NO'])
  key=len(room_list)
  alloted[key].append(room_list)



for i in range(4,1,-1):
    for j in range(len(alloted[i])):
        if alloted[i][j]!=None:
            dfinal.append(dftry[dftry['REGISTER NUMBER'].isin(alloted[i][j])].reset_index(drop=True))
        else:
            dfinal.append(pd.DataFrame())


unrec_hour=[]

for k in range(len(dfinal)):
    if dfinal[k].empty:
        continue
    totalh=11*5
    for i in dfinal[k].columns:
        if i==('REGISTER NUMBER',''):
            continue
        valid=0
        sum=0
        for j in range(0,len(dfinal[k])):
            sum += dfinal[k].loc[j,i]
        valid=(int) (sum/len(dfinal[k]))
        if(valid==1):
            totalh=totalh-1    #2/5
    unrec_hour.append(totalh)
    
    
xaxis1=[i for i in range(0,len(rec_hour))]
xaxis2=[i for i in range(0,len(unrec_hour))]

plt.plot(xaxis1,rec_hour,label='Load Hours after recom')
plt.plot(xaxis2,unrec_hour, label='Load Hours before recom')

plt.ylabel('Load Hours')
plt.ylim([25, 55])
plt.grid()
plt.legend()
plt.show()


print("Improvement is ")
print(str((mean(unrec_hour)-mean(rec_hour))/mean(unrec_hour)*100)+'%')

print("Mean Rec Hour" + str(mean(rec_hour)))
print("Mean Unrec hour" + str(mean(unrec_hour)))