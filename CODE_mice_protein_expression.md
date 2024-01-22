# Mice-protein-expression-ml-project
Machine Learning project from Ineuron . I have used SVM model for this project.

import numpy as np
import pandas as pd
df=pd.read_csv("Mice Protein Expression data.csv")

# Data processing.
t_colname_I=df[['ID','DYRK1A_N', 'ITSN1_N', 'BDNF_N', 'NR1_N', 'NR2A_N', 'pAKT_N', 'pBRAF_N', 'pCAMKII_N', 'pCREB_N', 'pELK_N', 'pERK_N', 'pJNK_N', 'PKCA_N', 'pMEK_N', 'pNR1_N', 'pNR2A_N', 'pNR2B_N', 'pPKCAB_N', 'pRSK_N', 'AKT_N', 'BRAF_N', 'CAMKII_N', 'CREB_N', 'ELK_N', 'ERK_N', 'GSK3B_N', 'JNK_N', 'MEK_N', 'TRKA_N', 'RSK_N', 'APP_N', 'Bcatenin_N', 'SOD1_N', 'MTOR_N', 'P38_N', 'pMTOR_N', 'DSCR1_N', 'AMPKA_N', 'NR2B_N', 'pNUMB_N', 'RAPTOR_N', 'TIAM1_N', 'pP70S6_N', 'NUMB_N', 'P70S6_N', 'pGSK3B_N', 'pPKCG_N', 'CDK5_N', 'S6_N', 'ADARB1_N', 'AcetylH3K9_N', 'RRP1_N', 'BAX_N', 'ARC_N', 'ERBB4_N', 'nNOS_N', 'Tau_N', 'GFAP_N', 'GluR3_N', 'GluR4_N', 'IL1B_N', 'P3525_N', 'pCASP9_N', 'PSD95_N', 'SNCA_N', 'Ubiquitin_N', 'pGSK3B_Tyr216_N', 'SHH_N', 'BAD_N', 'BCL2_N', 'pS6_N', 'pCFOS_N', 'SYP_N', 'H3AcK18_N', 'EGR1_N', 'H3MeK4_N', 'CaNA_N'
]].values
from sklearn.impute import SimpleImputer as sI
imputer=sI(missing_values=np.NaN,strategy='median')#np is used because we are using data in array formate
imputer=imputer.fit(t_colname_I[:,0:80])
t_colname_I[:,0:80]=imputer.transform(t_colname_I[:,0:80])
t_colname_I
df_1_I=pd.DataFrame(t_colname_I,columns=['ID','DYRK1A_N', 'ITSN1_N', 'BDNF_N', 'NR1_N', 'NR2A_N', 'pAKT_N', 'pBRAF_N', 'pCAMKII_N', 'pCREB_N', 'pELK_N', 'pERK_N', 'pJNK_N', 'PKCA_N', 'pMEK_N', 'pNR1_N', 'pNR2A_N', 'pNR2B_N', 'pPKCAB_N', 'pRSK_N', 'AKT_N', 'BRAF_N', 'CAMKII_N', 'CREB_N', 'ELK_N', 'ERK_N', 'GSK3B_N', 'JNK_N', 'MEK_N', 'TRKA_N', 'RSK_N', 'APP_N', 'Bcatenin_N', 'SOD1_N', 'MTOR_N', 'P38_N', 'pMTOR_N', 'DSCR1_N', 'AMPKA_N', 'NR2B_N', 'pNUMB_N', 'RAPTOR_N', 'TIAM1_N', 'pP70S6_N', 'NUMB_N', 'P70S6_N', 'pGSK3B_N', 'pPKCG_N', 'CDK5_N', 'S6_N', 'ADARB1_N', 'AcetylH3K9_N', 'RRP1_N', 'BAX_N', 'ARC_N', 'ERBB4_N', 'nNOS_N', 'Tau_N', 'GFAP_N', 'GluR3_N', 'GluR4_N', 'IL1B_N', 'P3525_N', 'pCASP9_N', 'PSD95_N', 'SNCA_N', 'Ubiquitin_N', 'pGSK3B_Tyr216_N', 'SHH_N', 'BAD_N', 'BCL2_N', 'pS6_N', 'pCFOS_N', 'SYP_N', 'H3AcK18_N', 'EGR1_N', 'H3MeK4_N', 'CaNA_N'
])
#df_1_I.info()
t_colname_D=df[['Genotype','Treatment','Behavior','class']].values
from sklearn.preprocessing import LabelEncoder as LE #to change common names to int values
LE_t_colname_D=LE()
i=0
while i<=3:
    t_colname_D[:,i]=LE_t_colname_D.fit_transform(t_colname_D[:,i])
    i=i+1
df_2_D=pd.DataFrame(t_colname_D,columns=['Genotype','Treatment','Behavior','class'])
df_2_D=df_2_D.astype({'Genotype':'int','Treatment':'int','Behavior':'int','class':'int'})
df_tmp1=df_2_D
df_tmp1=df_tmp1.drop(['class','Behavior'],axis=1)
df_2_D=df_2_D.drop(['Genotype','Treatment','Behavior','class'],axis=1)
df_2_D=df.iloc[:,80]
df_3_D=df.iloc[:,81]
#df_3_D.info()
#to add colname 'Genotype','Treatment','Behavior'to independent dataframe

df_1_I=pd.concat([df_1_I,df_tmp1],axis=1)
#df_1_I.info()

# TO SPLIT THE DATA SET for dependent 1 "Behavior".

from sklearn.model_selection import train_test_split #TO SPLIT THE DATA SET 
I_train,I_test,D1_train,D1_test=train_test_split(df_1_I,df_2_D,test_size=0.2,random_state=0)
#D1_train
#model training using support vector ml algorithm.
from sklearn.svm import SVC
model=SVC(kernel='linear')
model.fit(I_train,D1_train)

# To predict the output_1 "CLASS".

df_fopt1=model.predict(I_test)
model.score(I_test,D1_test)
df_fopt10=pd.DataFrame(df_fopt1,columns=['Behavior'])
#df_fopt1.to_csv(r'D:\pythonn alll\MICE_PROTEIN_EXPRESSION_2\df_fopt1.csv', index=False)
#df_fopt1

# To predict the output_2 "CLASS".

from sklearn.model_selection import train_test_split #TO SPLIT THE DATA SET 
I_train,I_test,D2_train,D2_test=train_test_split(df_1_I,df_3_D,test_size=0.2,random_state=0)
#D1_train
#model training using support vector ml algorithm.
from sklearn.svm import SVC
model=SVC(kernel='linear')
model.fit(I_train,D2_train)

# To predict the output_1 "CLASS",

df_fopt1=model.predict(I_test)
model.score(I_test,D2_test)
df_fopt20=pd.DataFrame(df_fopt2,columns=['class'])
#df_fopt1.to_csv(r'D:\pythonn alll\MICE_PROTEIN_EXPRESSION_2\df_fopt1.csv', index=False)
concatenated_df = pd.concat([df_fopt10, df_fopt20], axis=1)
concatenated_df
