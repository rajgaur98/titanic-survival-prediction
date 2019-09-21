import numpy as np
import scipy.optimize as op
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing

df = pd.read_csv('train.csv')
df.fillna(-99999, inplace=True)
df.drop(['PassengerId', 'Name'], axis = 1, inplace = True)
df.replace('male', 1, inplace = True)
df.replace('female', 0, inplace = True)

def pre_pro(df):
	word_list1 = []
	word_list2 = []
	word_list3 = []
	n_ticket = 0
	n_cabin = 0
	n_embarked = 0
	for ind in df.index:
		val = df['Ticket'][ind]
		if val == -99999:
			continue
		if val not in word_list1:
			word_list1.append(val)
			df['Ticket'][ind] = n_ticket
			n_ticket += 1
		else:
			x = word_list1.index(val)
			df['Ticket'][ind] = x
			
	for ind in df.index:
		val = df['Cabin'][ind]
		if val == -99999:
			continue
		if val not in word_list2:
			word_list2.append(val)
			df['Cabin'][ind] = n_cabin
			n_cabin += 1
		else:
			x = word_list2.index(val)
			df['Cabin'][ind] = x
			
	for ind in df.index:
		val = df['Embarked'][ind]
		if val == -99999:
			continue
		if val not in word_list3:
			word_list3.append(val)
			df['Embarked'][ind] = n_embarked
			n_embarked += 1
		else:
			x = word_list3.index(val)
			df['Embarked'][ind] = x
	return df
			
df = pre_pro(df)

X_train = np.array(df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch','Ticket', 'Fare', 'Cabin', 'Embarked']])
m, n = np.shape(X_train)
print(X_train[:10][:])
#X_train = np.hstack((np.ones((m,1)), X_train ))
X_train = preprocessing.scale(X_train)
y_train = np.array(df[['Survived']])


df1 = pd.read_csv('test.csv')
df1.fillna(-99999, inplace=True)
df1.drop(['PassengerId', 'Name'], axis = 1, inplace = True)
df1.replace('male', 1, inplace = True)
df1.replace('female', 0, inplace = True)
df1 = pre_pro(df1)
X_test = np.array(df1[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']])
i, j = np.shape(X_test)
#X_test = np.hstack((np.ones((i,1)), X_test ))
X_test = preprocessing.scale(X_test)

clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
clf.fit(X_train, y_train)
y_test = clf.predict(X_test)
y_test = y_test.reshape(i, )
print(clf.score(X_train, y_train))

passenger_id = [i for i in range(892,1310)]
pred = {'PassengerId': passenger_id,
		'Survived': y_test}

df2 = pd.DataFrame(pred, columns = ['PassengerId', 'Survived'])
export_csv = df2.to_csv(r'output.csv', index = None, header = True)

