# -*- coding: utf-8 -*-

import pandas
import numpy
base = pandas.read_csv('Pre-processing/pre-processing-credit-data/credit_data.csv')
#estatísticas da base
#print(base.describe())

#verificar valores negativos
### base.loc[base['age'] < 0])

#apagar a coluna com registros negativos
### base.drop('age', 1, inplace=True)

#apagar registros negativos
### base.drop(base[base.age < 0].index, inplace=True)

#preencher valores negativos com média
### base.loc[base.age < 0, 'age'] = base['age'][base.age > 0].mean()

#verificar campos com valores NaN
### base.loc[pandas.isnull(base['age'])]

#divisão entre previsores e classes
previsores = base.iloc[:,1:4].values
classe = base.iloc[:,4].values

#substituição de valores faltantes (NaN)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=numpy.nan, strategy='mean')
imputer = imputer.fit(previsores[:, 0:3])
previsores[:, 0:3] = imputer.transform(previsores[:, 0:3])

#escalonamento
#-----padronização
#-----x = (x - média(x))/desvio padrão(x)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)