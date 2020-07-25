import pandas as pd
import numpy as np
import datetime
import functools


df9=pd.read_csv("xxxxxxx/xx/xx/xxxxx/xxxxx/hdfc.csv",skiprows=1)##meantion the path


df9.columns = df9.columns.str.replace('/','')
df9.columns = df9.columns.str.replace('*','')
df9.columns = df9.columns.str.replace('.', '')
df9.columns = df9.columns.str.replace(' ', '')
df2=df9.rename(columns= {'Date':'Txn_Date','Narration':'Narration','ChqRefNo':'ChqRefNo','ValueDate':'ValueDate','WithdrawalAmount':'WithdrawalAmount','DepositAmount':'DepositAmount','ClosingBalance':'ClosingBalance'})
# #df2 = df.rename(columns={'Date':'Txn_date'},inplace=True)
#print(df2.columns)


df3 = df2.dropna(axis=0, subset=['ClosingBalance'])


#print(df3)
# #print(df3.to_csv("gh3.csv",index=False))

df3['ClosingBalance'] =df3['ClosingBalance'].str.replace(',?' , '')
df3['WithdrawalAmount'] =df3['WithdrawalAmount'].str.replace(',?' , '')
df3['DepositAmount'] =df3['DepositAmount'].str.replace(',?' , '')


# df3['Date'] = pd.to_datetime(df3['Txn_Date'], errors='coerce')
# df3['Txn_Date'] = pd.to_datetime(df3['Txn_Date'], format='%d/%m/%Y').dt.date
df3["Date"] = pd.to_datetime(df3["Txn_Date"], errors='coerce').dt.strftime('%m-%d-%y')

df3['Date'] = pd.to_datetime(df3['Txn_Date'], dayfirst=True)




df3["month"]=df3["Date"].dt.month
df3["year"]=df3["Date"].dt.year
df3["day"]=df3["Date"].dt.day

df3["month_year"] = pd.to_datetime(df3['Date'], errors='coerce').dt.to_period('M')


df4 = pd.to_numeric(df3['ClosingBalance'])


data5 = df3[['month_year','ClosingBalance']]
data6 = data5.dropna()
ds = data6['ClosingBalance'].astype(str).astype(float)
ClosingBalance = data6.assign(ClosingBalance=ds)
# print(ClosingBalance.groupby('month_year').describe())
# print(ClosingBalance.groupby('month_year').sum())


df32=ClosingBalance.groupby('month_year').describe()
df33=ClosingBalance.groupby('month_year').sum()
df30=pd.merge(df32,df33,on="month_year",left_index=False)


df30.rename(columns = {('ClosingBalance','count'):'Closing_Balnace_Count'}, inplace = True) 
df30.rename(columns = {('ClosingBalance','mean'):'Closing_Balnace_Average'}, inplace = True) 
df30.rename(columns = {'ClosingBalance':'Closing_Balnace'}, inplace = True) 

df30 = df30.reset_index()


df35=df30.drop(df30.columns[[3,4,5,6,7,8]], axis = 1)


data5 = df3[['month_year','WithdrawalAmount']]
data6 = data5.dropna()
ds = data6['WithdrawalAmount'].astype(str).astype(float)
WithdrawalAmount = data6.assign(WithdrawalAmount=ds)
# print(WithdrawalAmount.groupby('month_year').describe())
# print(WithdrawalAmount.groupby('month_year').sum())

df22=WithdrawalAmount.groupby('month_year').describe()
df23=WithdrawalAmount.groupby('month_year').sum()
df20=pd.merge(df22,df23,on="month_year",left_index=False)
# print(df20)
df20.rename(columns = {('WithdrawalAmount', 'count'):'Withdrawal_Amount_Count'}, inplace = True) 
df20.rename(columns = {('WithdrawalAmount', 'mean'):'WithdrawalAmount_Average'}, inplace = True) 
df20.rename(columns = {'WithdrawalAmount':'Withdrawal_Amount'}, inplace = True) 
df20 = df20.reset_index()
df25=df20.drop(df20.columns[[3,4,5,6,7,8]], axis = 1)

# print(df25)





data5 = df3[['month_year','DepositAmount']]
data6 = data5.dropna()
ds = data6['DepositAmount'].astype(str).astype(float)
DepositAmount = data6.assign(DepositAmount=ds)
# print(DepositAmount.groupby('month_year').describe())
# print(DepositAmount.groupby('month_year').sum())


df83=DepositAmount.groupby('month_year').describe()

df84=DepositAmount.groupby('month_year').sum()

df52=pd.merge(df83,df84,on="month_year",left_index=False)




df52.rename(columns = {('DepositAmount', 'count'):'Deposit_Amount_count'}, inplace = True) 
df52.rename(columns = {('DepositAmount', 'mean'):'Deposit_Amount_Average'}, inplace = True) 
df52.rename(columns = {'DepositAmount':'Deposit_Amount'}, inplace = True) 
df52 = df52.reset_index()

df55=df52.drop(df52.columns[[3,4,5,6,7,8]], axis = 1)

dfs = [df35, df25, df55]
df65_final = functools.reduce(lambda left,right: pd.merge(left,right,on='month_year'), dfs)

df = df3

initial=1
min_year=min(df['year'])
d = df.loc[(df['year'] == min_year),['month']]
min_month=min(d['month'])

max_year=max(df['year'])
d= df.loc[(df['year'] == max_year),['month']]
max_month=max(d['month'])

df['status'] =0
start_year = min_year
end_year = max_year
start_month = min_month
end_month = max_month

flag=0
while (start_year <=end_year):
    if flag == 0:
        start_month =min_month
    else:
        start_month =1
    while (start_month <=12):
            flag=1
            df.loc[(df["month"]== start_month)  & (df['year'] == start_year),["status"]] =initial
            start_month = start_month+1
            initial = initial + 1
    start_year = start_year+1

j1= df['status'].tail(1)
j=int(j1+1)
start = min(df['status'])
# df.to_csv("ffff.csv")


p=1
q=10
r=20

a_final = pd.DataFrame(columns=['Txn_Date','Narration','ChqRefNo','ValueDate','WithdrawalAmount','DepositAmount','ClosingBalance','Date',
                                'month','year','day','month_year'])
b_final = pd.DataFrame(columns=['Txn_Date','Narration','ChqRefNo','ValueDate','WithdrawalAmount','DepositAmount','ClosingBalance','Date',
                                'month','year','day','month_year'])
c_final = pd.DataFrame(columns=['Txn_Date','Narration','ChqRefNo','ValueDate','WithdrawalAmount','DepositAmount','ClosingBalance','Date',
                                'month','year','day','month_year'])
for i in range(1,j):
    a=0
    b=0
    p=1
    q=10
    r=20
    if i == start:
        a = df[((df['status']==i) & (df['day'] == p))].tail(1)
        if len(a)==0:
            a=df[((df['status']==i))].head(1)
    else:
        a = df[((df['status']==i) & (df['day'] == p))].tail(1)
        if len(a)==0:
            a = df[(df['status']==i-1)].tail(1)
           
    a_final = pd.concat([a, a_final], ignore_index=True)
       
    fg=0       
    b = df.loc[((df['status']==i) & (df['day'] == q))].tail(1)
    while (len(b)==0) & (fg==0):
        b = df[((df['status']==i) &(df['day']==max((df['day'])[(df['day']==q)])))].tail(1)
        q = q-1
        if (q != 0) :
            continue
        elif (len(b)==0):
            b = df[(df['status']==i-1)].tail(1)
               
           
         
    b_final = pd.concat([b, b_final], ignore_index=True)
      
       
    fg1=0       
    c = df.loc[((df['status']==i) & (df['day'] == r))].tail(1)
    while (len(c)==0) & (fg1==0):
        c = df[((df['status']==i) &(df['day']==max((df['day'])[(df['day']==r)])))].tail(1)
        r = r-1
        if (r != 0) :
            continue
        elif (len(c)==0):
            c = df[(df['status']==i-1)].tail(1)
               
           
         
    c_final = pd.concat([c, c_final], ignore_index=True)   
     
    
final_final = pd.concat([a_final,b_final, c_final], ignore_index=True)
final_final=final_final.sort_values(by=['month_year','day'])
# print(final_final)




final_final = final_final.reset_index(drop=True)
final1 = final_final.reset_index(drop=False)
final2 = final1.drop(['index'], axis = 1) 
final2.reset_index(inplace=True)
groups = final2['ClosingBalance'].groupby(np.arange(len(final2.index))//3)
# print(groups.describe())


aa = final2.reset_index()
 
 
index1 =pd.DataFrame(aa['index'].iloc[np.arange(len(aa)/3).repeat(3)])
index1 = index1.rename(columns={"index": "actl_index"})
 
aa1 = aa.assign(index1=index1.values)
aa1.set_index('index1', inplace=True)
aa2 = pd.DataFrame(aa1.groupby(['index1'])['ClosingBalance'].apply(lambda x: ','.join(x.astype(str))).reset_index())

 
aa2[['1st_Bal','10th_Bal','20th_Bal']] = aa2.ClosingBalance.str.split(",",expand=True,)
# aa2[['1st_Bal','10th_Bal','20th_Bal']] = aa2.ClosingBalance.str.split(",",expand=True,)
aa3 = aa2.iloc[:,2:]
# print(aa3)

df65_final[['1st_Bal','10th_Bal','20th_Bal']] = aa3

print(df65_final.reset_index())

df65_final.to_csv("final123.csv")


