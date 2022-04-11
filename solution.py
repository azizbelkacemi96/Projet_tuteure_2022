
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier


# lire l'ensemble de données
train_data = pd.read_csv('dataset/train.csv')
test_data = pd.read_csv('dataset/test.csv')


# sélectionner les fonctionnalités pour la formation
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
x_train = train_data[features]
x_test = test_data[features]

y_train = train_data['Survived']

# # vérifier les valeurs manquantes
print ('Informations sur les données dentraînement :')
#x_train.info()
print ('ilhaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaam\n',x_train)
print ('-'*30)
print ("Informations sur les données de test :")
#x_test.info()
print ('ilhaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaam\n',x_test)


# Utilisez le port avec le plus de connexions pour remplir la valeur nan du port de connexion
print ('\n\n\nInformations sur le port de connexion :')
print (x_train['Embarked'].value_counts())
x_train['Embarked'].fillna('S', inplace=True)
x_test['Embarked'].fillna('S', inplace=True)

# Utiliser l'âge moyen pour remplir les valeurs nan en âge
x_train['Age'].fillna(x_train['Age'].mean(), inplace=True)
x_test['Age'].fillna(x_test['Age'].mean(), inplace=True)

# Utiliser la moyenne du tarif pour renseigner les valeurs nan dans le tarif
x_test['Fare'].fillna(x_test['Fare'].mean(), inplace=True)

# Convertir les valeurs propres en vecteurs propres
dvec = DictVectorizer(sparse=False)

x_train = dvec.fit_transform(x_train.to_dict(orient='record'))
x_test = dvec.transform(x_test.to_dict(orient='record'))

# format vectoriel de fonction d'impression
print ('\n\n\n format de vecteur propre')
print (dvec.feature_names_)

# Soutenir les machines vectorielles
svc = SVC()
# arbre de décision
dtc = DecisionTreeClassifier()
# arbre de décision
rfc = RandomForestClassifier()
# régression logistique
# lr = LogisticRegression()
# Bayésien
nb = MultinomialNB()
# K est adjacent
knn = KNeighborsClassifier()
# AdaBoost
boost = AdaBoostClassifier()


print ('\n\n\nValidation du modèle:')
print ('SVM acc is', np.mean(cross_val_score(svc, x_train, y_train, cv=10)))
print ('DecisionTree acc is', np.mean(cross_val_score(dtc, x_train, y_train, cv=10)))
print ('RandomForest acc is', np.mean(cross_val_score(rfc, x_train, y_train, cv=10)))
# print ('LogisticRegression acc is', np.mean(cross_val_score(lr, x_train, y_train, cv=10)))
print ('NaiveBayes acc is', np.mean(cross_val_score(nb, x_train, y_train, cv=10)))
print ('KNN acc is', np.mean(cross_val_score(knn, x_train, y_train, cv=10)))
print ('AdaBoost acc is', np.mean(cross_val_score(boost, x_train, y_train, cv=10)))

# train
boost.fit(x_train, y_train)
# prédire
y_predict = boost.predict(x_test)
# enregistrer les résultats
result = {'PassengerId': test_data['PassengerId'],
          'Survived': y_predict}
result = pd.DataFrame(result)
result.to_csv('submission.csv',index=False)
