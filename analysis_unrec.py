import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cufflinks as cf
import pickle
import math

def sigm(row):
    for i,item in enumerate(row):
        row[i]=1/(1+math.exp(-3*row[i]))
    return row

# Create an empty dataframe with no columns or rows
df = {'Monday':pd.DataFrame(),'Tuesday':pd.DataFrame(),'Wednesday':pd.DataFrame(),'Thursday':pd.DataFrame(),'Friday':pd.DataFrame()}


for j in df.keys():
    col=['Name']
    for i in range(8,20):
        col.append(str(i))
    df[j] = pd.DataFrame(columns=col, dtype=float)


df4=pd.read_csv('C:/Users/prana/Downloads/4realtarp.csv')
df3=pd.read_csv('C:/Users/prana/Downloads/3realtarp.csv')
df2=pd.read_csv('C:/Users/prana/Downloads/2realtarp.csv')


df4=pd.read_csv('C:/Users/prana/Downloads/44realtarp_new.csv')
df4=df4.fillna(0)
df4

'''
totalh=11

df=[df4,df3,df2]
kwh=0

misc_kwh = 0.116*2*10*20


for k in range(len(df)):
    totalh=11
    for i in range(1,len(df[k].columns)):
        valid=0
        sum=0
        for j in range(0,len(df[k])):
            sum+=df[k].iloc[j,i]
        valid=(int) (sum/len(df[k]))
        if(valid==1):
            totalh=totalh-1    #2/5
            
    print(totalh)
    if k==0:
        tmp=0.200*totalh*20+misc_kwh*4
    elif k==1:
        tmp=0.12*totalh*20+misc_kwh*3
    else:
        tmp=0.1*totalh*20+misc_kwh*2
    kwh+=tmp
    

kbi=kwh*2

if(kbi<=200):
    cost=0
elif(kbi>200 and kbi<=400):
    cost=200*0+(kbi-200)*4.5
elif(kbi>400 and kbi<=500):
    cost=200*0+200*4.5+(kbi-400)*6
elif(kbi>500 and kbi<=600):
    cost=200*0+200*4.5+100*6+(kbi-500)*8
elif(kbi>600 and kbi<=800):
    cost=200*0+200*4.5+100*6+100*8+(kbi-600)*9
elif(kbi>800 and kbi<=1000):
    cost=200*0+200*4.5+100*6+100*8+200*9+(kbi-800)*10
else:
    cost=200*0+200*4.5+100*6+100*8+200*9+200*10+(kbi-1000)*11
    


print(kwh)
print(cost)
print(cost/6)
'''


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
df

ind2=[['Monday','Monday','Monday','Monday','Monday','Monday','Monday','Monday','Monday','Monday','Monday'],
      ['8','9','10','11','12','13','14','15','16','17','18']]

col2=['Pranayyy','Pranay2','Pranay3']
data_tt2=[[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0],[0,1,0]]
dftry=pd.DataFrame(data_tt2,index=ind2,columns=col2)
dftry=dftry.T

for i in range(len(df4)):
    for k in data_days:
      for j in df4.columns:
        
        if(j=='Name'):
            continue
        
        if(df4.iloc[i][j]==0):
            continue
        
        try:
          df.loc[(df4.loc[i,'Name'],k),map[k][j]]=1
          dftry.loc[df4.loc[i,'Name'],(k,map[k][j])]=1
        except:
          print(j)

df=df.fillna(0)
dftry=dftry.fillna(0)
df=df[5:]
dftry=dftry[3:]
dftry['Name']=dftry.index
dftry['Name']=dftry['Name'].astype(str)
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
feature_df=feature_df.drop(['Name'],axis=1)

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import neighbors

scaler = MinMaxScaler()
feature_arr = scaler.fit_transform(feature_df)

model = neighbors.NearestNeighbors(n_neighbors=4 , algorithm = 'ball_tree')
model.fit(feature_arr)

pickle.dump(model, open('model.pkl','wb'))
pickle.dump(feature_df, open('feature.pkl','wb'))
pickle.dump(dftry, open('df.pkl','wb'))

dist_mat , id_list_mat = model.kneighbors(feature_arr)

#Recommending Mates
def rec_mates(name):
    r_list = []
    match_list=[]
    r_idx = dftry[dftry['Name'] == name].index
    r_idx = r_idx[0] #iff multiple index found
    count=0
    for num in id_list_mat[r_idx]:
        if((dftry.iloc[num].Name[0]!=name) and (dftry.iloc[num].Avail!=1)[0] and ( (df2.loc[name][dftry.iloc[num].Name]>0.20)[0]  #or count<5
                                                                        )):
         r_list.append(dftry.iloc[num].Name[0])
         match_list.append(df2.loc[name][dftry.iloc[num].Name][0])
         count=count+1
    return r_list,match_list

flag=1
details={};


class Rooms:
  def __init__(self,nobeds1,nobeds2,nobeds3,nobeds4):
      self.details={0: [],
               1:list(range(101,101+nobeds1)),
               2:list(range(201,201+nobeds2)),
               3:list(range(301,301+nobeds3)),
               4:list(range(401,401+nobeds4))}
      
          
      
      
rooms = Rooms(1,1,2,1)
alloted={2:None,3:None,4:None}

while(flag==1):
    name=input("Enter your name : ")
    if(name not in list(dftry['Name'])):
        print("Wrong Name")
        continue
    #dell=['MS2','MS3']
    if(dftry.loc[dftry[dftry['Name']==name].index,'Avail'].iloc[0]==1):
        print("Already opted")
        print("Room alotted is : ",dftry.loc[dftry[dftry['Name']==name].index,'Rooms'].iloc[0])
        continue
    
    rec_,match_ = rec_mates(name)
    for i in range(0,len(rec_)):
        print(str(rec_[i]) + "       " + str(match_[i]))
    sel_list=input('Enter your desired roomates : ')
    sel_list=list(sel_list.split(","))

    
    
    check=False
    for i in sel_list:
        if i not in rec_:
            check=True

    if(len(sel_list)>3 or len(sel_list)>len(rec_) or check):
        print("Fraud")
        flag=1
        continue
    
    
    
    size=len(sel_list)+1
    room=int(input('No. Of Beds'))
    
    
    if(size>room):
        print("Select Appropriate Room")
        continue
    else:
        x=rooms.details[room].pop(0)
        dftry.loc[dftry[dftry['Name']==name].index,'Rooms']=x
        rooms.details[room-size].append(x)
        print("Room Granted",x)
    

    for i in sel_list:
        dftry.loc[dftry[dftry['Name']==i].index,'Avail']=1 
        dftry.loc[dftry[dftry['Name']==i].index,'Rooms']=x
        
        
    dftry.loc[dftry[dftry['Name']==name].index,'Avail']=1
    
    
    #rec_.remove(name)
    #rec_ = [ele for ele in rec_ if ele not in dell]
    details[name]=sel_list
    print("Successful")
    
    sel_list.append(name)
    alloted[room]=sel_list
    
    
    
    flag=int(input("Flag : "))
    
   
dfinal=[]


for i in range(4,1,-1):
    if alloted[i]!=None:
        dfinal.append(dftry[dftry['Name'].isin(alloted[i])].reset_index(drop=True))
    else:
        dfinal.append(pd.DataFrame())


for k in range(len(dfinal)):
    if dfinal[k].empty:
        continue
    totalh=11*5
    for i in dfinal[k].columns:
        if i==('Name',''):
            continue
        valid=0
        sum=0
        for j in range(0,len(dfinal[k])):
            sum += dfinal[k].loc[j,i]
        valid=(int) (sum/len(dfinal[k]))
        if(valid==1):
            totalh=totalh-1    #2/5
    print(totalh)
    


f = plt.figure()
f.set_figwidth(15)
f.set_figheight(9)

sns.heatmap(df2,annot=True,cmap="coolwarm_r")


plt.show()





'''
for i in range(len(df2)):
    
    for j in df2.columns:
        
        if(j=='Name'):
            continue
        
        if(df2.iloc[i][j]==0):
            continue
        
        for w in ['Monday','Tuesday','Wednesday','Thursday','Friday']:
            try:
                ind = map[w][j]
                if(len(df[w][df[w]['Name']==df2.iloc[i]['Name']])==0):
                    
                else:
                    df[w][df[w]['Name']==df2.iloc[i]['Name']][ind]=1
                
            except:
                continue
            
            
#df.loc[len(df.index)] = []
'''

'''

import numpy as np
import xarray as xr

#make this example reproducible
np.random.seed(1)

#create 3D dataset
xarray_3d = xr.Dataset(
    {"product_A": (("year", "quarter"), np.random.randn(2, 4))},
    coords={
        "year": [2021, 2022],
        "quarter": ["Q1", "Q2", "Q3", "Q4"],
        "product_B": ("year", np.random.randn(2)),
        "product_C": 50,
    },
)

#view 3D dataset
print(xarray_3d)

df_3d = xarray_3d.to_dataframe()

#view 3D DataFrame
print(df_3d)

'''