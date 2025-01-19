# Pr-diction-des-prix-des-maisons-au-USA
Projet Machine Learning avec Jupyter Notebook

### Prédiction du prix des maisons en USA 

#### Business Objective 
- Maximiser les profits des investisseurs, Réduire les risques des propriétés sous-évaluées et Optimiser les décisions d'achat et de vente.
- Identifier les états "STATE" les plus rentables pour les investissements immobiliers.
#### Data Science Objective
#### 1.	Maximiser les profits des investisseurs, Réduire les risques en identifiant les propriétés sous-évaluées et optimiser les décisions d'achat et de vente.
Prédire les cadres de prise de décision, Regression techniques des anomalies et détection algorithmiques pour optimiser l'achat et la vente
#### 2.	Identifier les états "STATE" les plus rentables pour les investissements immobiliers.
Classfication par état "STATE"
### Description data
#### Les importations
import pandas as pd  # Manipuler les données de type DataFrame
import numpy as np  # Manipuler les tableaux
import matplotlib.pyplot as plt  # Faire des graphiques
import seaborn as sns  # Manipuler les visualisations
from sklearn.model_selection import train_test_split  # Diviser les données
from sklearn.linear_model import LinearRegression  # Modèle de régression linéaire
from sklearn import metrics  # Évaluation des métriques
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import statsmodels.api as sm  # Import statsmodels
import flask, sklearn, pandas

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import mean_squared_error, r2_score
from imblearn.over_sampling import SMOTE

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import statsmodels.api as sm

import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier  # Exemple avec un modèle de forêt aléatoire

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

from sklearn import svm
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt  # Assurez-vous d'importer matplotlib pour l'affichage
### Récuperation des données
# Charger les données
data = pd.read_csv('USA_Housing.csv', sep=';', header=0)
# Chemin du fichier
import os
print(os.getcwd())
# Afficher le nbr de colonne et de ligne
print(data.shape)   #data.shape()
# Description de la data
data.describe()     #print(data.describe)   
•	Revenu moyen de la zone : Le revenu moyen est d'environ 68 583 $ avec un écart-type de 10 658 $. Le revenu minimum est de 17 796 $ et le maximum de 107 701 $.
•	Âge moyen des maisons : L'âge des maisons varie de 2,6 à 9,5 ans avec une moyenne d'environ 6 ans.
•	Nombre moyen de pièces : Les maisons comptent entre 3,2 et 10,7 pièces, avec une moyenne de 6,99 pièces.
•	Nombre moyen de chambres : Le nombre moyen de chambres est d'environ 4, avec une variation allant de 2 à 6,5 chambres.
•	Population de la zone : La population varie considérablement, allant de 173 personnes à plus de 69 000, avec une moyenne d'environ 36 163 personnes.
•	Prix : Les prix des maisons varient de 15 939 $ à 2,47 millions $, avec un prix moyen d'environ 1,23 million $
data.info()
data.head()
# Vérification des doublons
data.duplicated().sum()
# Créer une copie des données originales pour garder l'historique
data_with_duplicates = data.copy()
# Vérifier les dimensions des deux DataFrames pour s'assurer qu'ils sont identiques
print(f"Taille de data originale: {data.shape}")
print(f"Taille de la copie data_with_duplicates : {data_with_duplicates.shape}")
# Supprimer les doublons et créer une nouvelle base de données sans doublons
data_without_duplicates = data.drop_duplicates()
print(f"Les doublons sont supprimés et la la base de données sans doublons est créer")
# Vérification des dimensions après suppression des doublons
print(f"Taille de data_without_duplicates est : {data_without_duplicates.shape}")
print(f"Doublons supprimés : {data_with_duplicates.shape[0] - data_without_duplicates.shape[0]}")
#### On a supprimés les doublons qui peuvent fausser les résultats de certaines observations
# Valeur manquantes
data.isnull().sum()
#### * Le résultat indique qu'on a un valeur manquant pour tout les colonnes sauf la colonne Address.
#### * Les valeurs manquantes peuvent affecter l'analyse et les résultats du modèle, surtout si on utilises ces colonnes pour l'apprentissage automatique.
### Pour Traiter les lignes avec des valeurs manquantes on a plusieur possibilité:
#### 1- On peut supprimer les lignes avec des valeurs manquantes  =>  data_cleaned = data.dropna().

#### print(f"Taille après suppression des lignes avec des valeurs manquantes : {data_cleaned_dropna.shape}")
#### 2- Imputation avec la moyenne, la médiane ou le mode.
# Créer une copie des données originales pour garder l'historique
data_original = data_without_duplicates.copy()

# Vérification de la taille pour s'assurer que la copie est identique
print(f"Taille du dataset original : {data_original.shape}")
##### 2.1- Imputation avec la moyenne
from sklearn.impute import SimpleImputer
# Créer une base de données pour imputation
data_cleaned_imputed_moyenne = data_original.copy()

# Imputer avec la moyenne pour les colonnes numériques
imputer = SimpleImputer(strategy='mean')
data_cleaned_imputed_moyenne[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 
      'Avg. Area Number of Bedrooms', 'Area Population', 'Price']] = imputer.fit_transform(
    data_cleaned_imputed_moyenne[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 
                          'Avg. Area Number of Bedrooms', 'Area Population', 'Price']])
data_cleaned_imputed = data_cleaned_imputed_moyenne.copy()

# Vérification de la taille après imputation
print(f"Taille après imputation des valeurs manquantes : {data_cleaned_imputed_moyenne.shape}")
##### 2.2- Imputation avec la médiane (pour éviter l'influence des valeurs extrêmes)
from sklearn.impute import SimpleImputer
# Créer une base de données pour imputation
data_cleaned_imputed_mediane = data_original.copy()

# Appliquer l'imputeur sur les mêmes colonnes
imputer = SimpleImputer(strategy='median')
data_cleaned_imputed_mediane[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 
      'Avg. Area Number of Bedrooms', 'Area Population', 'Price']] = imputer.fit_transform(
    data_cleaned_imputed_mediane[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 
          'Avg. Area Number of Bedrooms', 'Area Population', 'Price']])
data_cleaned_imputed = data_cleaned_imputed_mediane.copy()

# Vérification de la taille après imputation
print(f"Taille après imputation des valeurs manquantes : {data_cleaned_imputed_mediane.shape}")
##### 2.3- Imputation par le mode
from sklearn.impute import SimpleImputer

# Créer une base de données pour imputation par le mode
data_cleaned_imputed_mode = data_original.copy()

# Créer un imputeur pour remplacer les valeurs manquantes par le mode
imputer_mode = SimpleImputer(strategy='most_frequent')

# Appliquer l'imputeur sur les colonnes numériques
data_cleaned_imputed_mode[['Avg. Area Income', 'Avg. Area House Age', 
                   'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 
                   'Area Population', 'Price']] = imputer_mode.fit_transform(
    data_cleaned_imputed_mode[['Avg. Area Income', 'Avg. Area House Age', 
                        'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 
                        'Area Population', 'Price']]
)
data_cleaned_imputed = data_cleaned_imputed_mode.copy()

# Vérification de la taille après imputation
print(f"Taille après imputation par le mode : {data_cleaned_imputed_mode.shape}")
##### 2.3- Remplacer par une valeur fixe
# Créer une base de données pour remplacement par une valeur fixe
data_cleaned_fixed = data_original.copy()

# Remplacer les valeurs manquantes par 0 ou une autre valeur fixe
data_cleaned_fixed.fillna(0, inplace=True)
data_cleaned_imputed = data_cleaned_fixed.copy()

# Vérification de la taille après remplacement
print(f"Taille après remplacement des valeurs manquantes : {data_cleaned_fixed.shape}")
### Créer une colonne indicatrice pour les valeurs manquantes
# Créer une base de données pour ajout de colonne indicatrice
data_with_indicators = data.copy()

# Créer une colonne indicatrice pour chaque variable avec des valeurs manquantes
data_with_indicators['Income_missing'] = data['Avg. Area Income'].isnull().astype(int)
data_with_indicators['House_Age_missing'] = data['Avg. Area House Age'].isnull().astype(int)
data_with_indicators['Rooms_missing'] = data['Avg. Area Number of Rooms'].isnull().astype(int)
data_with_indicators['Bedrooms_missing'] = data['Avg. Area Number of Bedrooms'].isnull().astype(int)
data_with_indicators['Population_missing'] = data['Area Population'].isnull().astype(int)

# Imputer les valeurs manquantes pour les colonnes numériques uniquement
numerical_columns = ['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 
                     'Avg. Area Number of Bedrooms', 'Area Population', 'Price']

data_with_indicators[numerical_columns] = data_with_indicators[numerical_columns].fillna(
    data[numerical_columns].mean())

# Vérification de la taille après ajout des indicateurs et traitement des valeurs manquantes
print(f"Taille après ajout des indicateurs et imputation : {data_with_indicators.shape}")
Le résultat indique que la taille du DataFrame après ajout des indicateurs et imputation est de (5012, 12). 
Cela signifie :

5012 : Il y a 5012 lignes dans le DataFrame, ce qui est le même nombre de lignes que dans les données originales.
Aucune ligne n'a été supprimée pendant le traitement des valeurs manquantes.

12 colonnes : Le nombre de colonnes est passé de 7 à 12. Cela s'explique par :

Les 7 colonnes originales (par exemple, Avg. Area Income, Avg. Area House Age, etc.).
L'ajout de 5 nouvelles colonnes indicatrices (Income_missing, House_Age_missing, Rooms_missing, Bedrooms_missing,
Population_missing), qui enregistrent les positions où les valeurs étaient manquantes.

Interprétation :
Les colonnes indicatrices : Elles permettent de savoir quelles valeurs étaient manquantes dans les colonnes 
numériques avant l'imputation. Chaque colonne indicatrice contient soit 1 (indiquant une valeur manquante), 
soit 0 (valeur non manquante).

Imputation : Les valeurs manquantes dans les colonnes numériques ont été remplacées par la moyenne des autres 
valeurs, ce qui permet de conserver l'intégrité des données sans perte d'information.
### Comparer les versions
# Comparer la taille des datasets
print(f"Taille de la base de données original : {data_original.shape}")
print(f"Taille après suppression des lignes avec des valeurs manquantes : {data_without_duplicates.shape}")
print(f"Taille après imputation des valeurs manquantes : {data_cleaned_imputed.shape}")
print(f"Taille après ajout des indicateurs et imputation : {data_with_indicators.shape}")
### Analyse data
# Afficher les correlations entre les variables 
sns.pairplot(data_cleaned_imputed)
### 1. Corrélations significatives :
Avg. Area Income et Price : On remarque un léger alignement des points en montée entre ces deux variables, 
indiquant une corrélation positive. Cela signifie que les revenus moyens dans une zone tendent à augmenter 
avec le prix des maisons.

Area Population et Price : La dispersion des points suggère qu'il n'y a pas de corrélation significative 
entre la population de la zone et le prix des maisons.

Avg. Area House Age et Price : Il n’y a pas de tendance claire entre l’âge moyen des maisons et leur prix. 
Les points sont dispersés de façon aléatoire, ce qui montre peu ou pas de lien.

### 2. Corrélations faibles ou absentes :
Avg. Area Number of Rooms et Price : Bien que la variable soit catégorique (de 4 à 10 pièces), il n'y a pas 
de tendance forte entre le nombre de pièces et le prix. Les points sont disposés en groupes horizontaux sans 
relation linéaire.

Avg. Area Number of Bedrooms et Price : Même constat, il y a peu de lien direct visible entre le nombre moyen 
de chambres et le prix. Les points sont alignés de manière horizontale et peu structurée.

### 3. Distributions des variables :
Les distributions univariées (sur la diagonale) montrent que certaines variables comme Avg. Area Income et 
Price suivent une distribution normale, avec des valeurs concentrées autour d'une moyenne.
Les variables comme Avg. Area Number of Rooms et Avg. Area Number of Bedrooms semblent suivre une distribution 
discrète ou catégorique, avec des valeurs spécifiques récurrentes (barres visibles).
Conclusion :
Relation prix/revenu : Il y a une légère corrélation positive entre le revenu moyen d'une zone et le prix des 
maisons.
Absence de corrélation pour d'autres variables : Certaines variables, comme le nombre de chambres ou l'âge moyen 
des maisons, semblent avoir peu d'influence directe sur le prix.
### Preparation data
x= data_cleaned_imputed [['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population']] 
y= data_cleaned_imputed ['Price']
### Préparer la base de données en Train et Test
# Définir X et y
X = data_cleaned_imputed.drop(['Price', 'Address'], axis=1)  # Supprimer la colonne 'Address' could not convert string to float # Toutes les colonnes sauf 'Price'
y = data_cleaned_imputed ['Price']  # La colonne 'Price' comme cible
# Split des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
# Vérifier les dimensions des ensembles de données
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
### Objectif des données en Train et Test:
Entraînement : Le modèle sera entraîné sur l'ensemble X_train et y_train, c'est-à-dire qu'il apprendra à établir des relations entre les caractéristiques (colonnes) et les cibles (prix, par exemple).

Test : L'ensemble X_test et y_test servira à tester la généralisation du modèle. 

Après l'entraînement, tu pourras prédire les valeurs cibles sur X_test et comparer ces prédictions avec les valeurs réelles dans y_test.
## Application du modele K-Nearest Neighbors KNN
##### Préparer les données
# Supposons que vous ayez déjà nettoyé les données
X = data_cleaned_imputed[['Avg. Area Income', 'Avg. Area House Age', 
                            'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 
                            'Area Population']]
y = data_cleaned_imputed['Price']

# Diviser les données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Vérifiez la taille des ensembles
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")
# Créer le modèle KNN et l'entraîner
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)
# Prédire sur les données de test
y_knn_pred = knn_model.predict(X_test)
# Calcul des erreurs pour le modèle KNN
mae_knn = mean_absolute_error(y_test, y_knn_pred)
mse_knn = mean_squared_error(y_test, y_knn_pred)
rmse_knn = np.sqrt(mse_knn)
r2_score_knn = knn_model.score(X_test, y_test)
# Afficher les résultats de KNN
print(f"MAE (KNN): {mae_knn}")
print(f"MSE (KNN): {mse_knn}")
print(f"RMSE (KNN): {rmse_knn}")
print(f"Score R² (KNN): {r2_score_knn}")
#### MAE l’erreur moyenne absolue, soit la différence moyenne entre les valeurs réelles et les valeurs prédites. Ici, en moyenne, les prédictions du modèle KNN s’écartent de 201,929.90 des prix réels, ce qui est assez élevé pour des valeurs de prix.
#### MSE pénalise les grandes erreurs plus fortement, puisqu’il élève les erreurs au carré.
#### RMSE (Root Mean Squared Error) En prenant la racine carrée du MSE, l’erreur de 254,994.07 montre encore une fois une différence assez importante
 #### La valeur 0.4827 Score R² signifie que le modèle explique environ 48.3% de la variance des prix de maisons. 
### Voyant le R² ajusté
# Calculer R² ajusté
n = X_test.shape[0]  # Nombre d'échantillons
p = X_test.shape[1]  # Nombre de caractéristiques
r2_adjusted_knn = 1 - (1 - r2_score_knn) * (n - 1) / (n - p - 1)
print(f"R² ajusté (KNN): {r2_adjusted_knn}")
#### R² ajusté de 0.4810, signifie que, même en prenant en compte le nombre de variables explicatives dans le modèle, celui-ci n'explique toujours qu'environ 48.1% de la variance des prix.
# --- Section Matrice de Confusion ---
# Créer une catégorie de prix
data_cleaned_imputed['Price_Category'] = pd.qcut(data_cleaned_imputed['Price'], q=3, labels=['Low', 'Medium', 'High'])

# Définir les variables pour la classification
X_class = data_cleaned_imputed[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms']]
y_class = data_cleaned_imputed['Price_Category']

# Diviser les données en ensemble d'entraînement et de test
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.2, random_state=42)

# Créer et entraîner le modèle KNN pour la classification
knn_model_class = KNeighborsClassifier(n_neighbors=5)
knn_model_class.fit(X_train_class, y_train_class)

# Prédire les classes pour l'ensemble de test
y_pred_class = knn_model_class.predict(X_test_class)

# Calculer et afficher la matrice de confusion
conf_matrix = confusion_matrix(y_test_class, y_pred_class, labels=['Low', 'Medium', 'High'])

# Visualiser la matrice de confusion
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
plt.xlabel("Prédictions")
plt.ylabel("Réel")
plt.title("Matrice de confusion pour KNN")
plt.show()

# Afficher le rapport de classification pour plus d'indicateurs
print(classification_report(y_test_class, y_pred_class))
#### Interprétation des résultats :
Low :

Précision : 0.54
Rappel : 0.63
F1-Score : 0.58
Support : 342
Medium :

Précision : 0.41
Rappel : 0.28
F1-Score : 0.33
Support : 335
High :

Précision : 0.57
Rappel : 0.66
F1-Score : 0.61
Support : 323
Global Metrics :

Exactitude : 0.52
Macro Avg : 0.51
Weighted Avg : 0.51
Conclusions :
La classe "High" a la meilleure performance, tandis que "Medium" montre une performance plus faible, surtout en termes de rappel et de F1-score.
L'exactitude globale du modèle est relativement modeste à 0.52, ce qui pourrait indiquer que le modèle a des difficultés à prédire certaines classes correctement.
Envisagez d'optimiser votre modèle ou d'explorer d'autres algorithmes si ces résultats ne répondent pas à vos attentes.
# --- Section Ordinary Least Squares ---
#####
Pour la partie OLS du modele KNN, il est nécéssaire de mentionner que: la régression par les moindres carrés ordinaires (Ordinary Least Squares, OLS) avec un modèle KNN, il est nécessaire de bien comprendre que l'OLS est généralement utilisé pour les modèles linéaires, comme ceux construits avec LinearRegression de scikit-learn. Le KNN est, lui, un algorithme non paramétrique et ne se prête pas directement à une analyse OLS car il n’a pas de coefficients à ajuster de manière linéaire..

Cependant, si on souhaite évaluer la précision de notre modèle KNN, nous pouvons utiliser les erreurs moyennes telles que Mean Absolute Error (MAE) ou Mean Squared Error (MSE) qui indiquent la précision du modèle sans ajuster des paramètres comme dans une OLS classique.
MAE (KNN): 201929.8975871893
MSE (KNN): 65021975821.54046
RMSE (KNN): 254994.07016936777
Score R² (KNN): 0.4827404380160554
## Application du modele Régression Lineaire
# Transformer les données en supprimant la colonne 'Address' et en utilisant pd.get_dummies
X = pd.get_dummies(data_cleaned_imputed.drop(['Price', 'Address'], axis=1), drop_first=True)
y = data_cleaned_imputed['Price']
# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
# Vérifier que toutes les colonnes sont numériques
print("Types de données dans X_train:")
print(X_train.dtypes)
# Entraîner le modèle
regressor = LinearRegression()
regressor.fit(X_train, y_train)  # Entraîner le modèle

print("Le modèle a été entraîné avec succès.")
### Objectif des données en Train et Test:
Entraînement : Le modèle sera entraîné sur l'ensemble X_train et y_train, c'est-à-dire qu'il apprendra à établir des relations entre les caractéristiques (colonnes) et les cibles (prix, par exemple).

Test : L'ensemble X_test et y_test servira à tester la généralisation du modèle. 

Après l'entraînement, tu pourras prédire les valeurs cibles sur X_test et comparer ces prédictions avec les valeurs réelles dans y_test.
### Construction du model
# Construction du modèle
regressor = LinearRegression()
regressor.fit(X_train, y_train)  # Entraîner le modèle

# Évaluation du modèle
print(f"Intercept : {regressor.intercept_}")
print(f"Coefficients : {regressor.coef_}")
### Interprétation :
Intercept : Il s'agit de la valeur prédite de la variable cible (par exemple, le prix de la maison) lorsque toutes les variables explicatives sont égales à 0.
C'est le point où la ligne de régression coupe l'axe des ordonnées (l'axe y) lorsque toutes les variables explicatives sont nulles.

Dans notre cas l'intercept est négatif : cela pourrait indiquer que sans aucune contribution des variables explicatives (quand toutes sont à zéro), le modèle prédit une valeur cible négative, ce qui n'est pas nécessairement réaliste dans un contexte comme les prix des maisons.

#### Cependant, les intercepts peuvent parfois être difficiles à interpréter directement, surtout si les valeurs nulles pour les variables explicatives n'ont pas de sens pratique. L'important est surtout la précision de notre modèle global.
# Créer un DataFrame pour les coefficients
coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
print(coeff_df)
### Chaque élément de la liste de coefficients correspond à une des variables explicatives dans le modèle.

2.16230248e+01 : Cela signifie que pour chaque unité d'augmentation de la première variable (par exemple, le revenu moyen de la zone), le prix de la maison augmente en moyenne de 21.62 unités, en gardant les autres variables constantes.

1.65209380e+05 : Pour chaque unité d'augmentation de la deuxième variable, le prix augmente de 165 209 unités.

1.21462297e+05, 1.22296313e+03, 1.52173463e+01 : Le même raisonnement s'applique à chacune des autres variables.

#### En résumé : Les coefficients montrent à quel point chaque variable explicative influence la variable cible (prix des maisons).
# Afficher le DataFrame
print(coeff_df)
### Interprétation des coefficients dans le DataFrame :
Avg. Area Income : 21.623025
Cela signifie que pour chaque augmentation d'une unité dans le revenu moyen de la zone, le prix de la maison augmente en moyenne de 21.62 unités, toutes les autres variables étant constantes.

Avg. Area House Age : 165209.379685
Une augmentation d'une unité dans l'âge moyen des maisons de la zone entraîne une augmentation moyenne de 165 209 unités dans le prix de la maison.

Avg. Area Number of Rooms : 121462.296674
Chaque pièce supplémentaire dans la maison est associée à une augmentation moyenne du prix de 121 462 unités.

Avg. Area Number of Bedrooms : 1222.963127
Chaque chambre supplémentaire entraîne une augmentation moyenne du prix de 1 222 unités.

Area Population : 15.217346
Pour chaque personne supplémentaire dans la population de la zone, le prix de la maison augmente en moyenne de 15.22 unités.

#### En résumé : Le coefficient le plus élevé est celui de l'âge moyen des maisons dans la zone (165 209 unités), ce qui suggère que cette variable a un impact beaucoup plus fort sur le prix des maisons par rapport aux autres variables. Les variables comme le nombre moyen de chambres et la population de la zone ont des coefficients relativement faibles, suggérant qu'elles ont moins d'influence sur le prix.
# Prédictions
y_predict = regressor.predict(X_test)
# Visualisation des résultats
plt.scatter(y_test, y_predict, color='blue', label='Prédictions')
plt.xlabel('Valeurs Réelles (y_test)')
plt.ylabel('Valeurs Prédites (y_predict)')
plt.title('Comparaison des Valeurs Réelles et Prédites')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='green', linestyle='--', label='y=x')  # Ligne y=x
plt.legend()
plt.show()
#### On a une bonne corrélation entre les valeurs réelles et les valeurs prédites, comme le montre l'alignement des points le long de la diagonale. Il pourrait y avoir quelques écarts, mais globalement, la performance semble correcte.
### Evaluation des Metrics de la regression
# Évaluation des métriques de régression
print('MAE:', metrics.mean_absolute_error(y_test, y_predict))
print('MSE:', metrics.mean_squared_error(y_test, y_predict))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_predict)))
print('R²:', metrics.r2_score(y_test, y_predict))
#### MAE (Mean Absolute Error) : 81,471.76

Le MAE représente l'erreur moyenne absolue entre les prédictions du modèle et les valeurs réelles. Ici, cela signifie que, en moyenne, notre modèle de régression prédit un prix qui diffère de la réalité d'environ 81,471.76 unités de la devise utilisée.
C'est une bonne mesure de l'erreur moyenne sans tenir compte de la direction (positive ou négative).
MSE (Mean Squared Error) : 10,241,977,764.37

Le MSE représente la moyenne des carrés des erreurs. Il pénalise davantage les grandes erreurs car il élève au carré la différence entre les prédictions et les valeurs réelles. Ici, le MSE est relativement élevé (en raison des unités au carré), mais il est utile pour détecter des erreurs importantes.
RMSE (Root Mean Squared Error) : 101,202.66

Le RMSE est simplement la racine carrée du MSE, et il exprime l'erreur en unités originales (comme le MAE). 
Dans notre cas, cela signifie que l'erreur quadratique moyenne de notre modèle est d'environ 101,202.66 unités. 
C'est une métrique souvent utilisée pour interpréter la performance du modèle en termes plus compréhensibles.
### Objectif avoir un R²_Score prés de 1
print('R²:', metrics.r2_score (y_test, y_predict))
R² (Coefficient de détermination) : 0.9179
mesure la proportion de la variance dans la variable dépendante (dans ce cas, les prix des maisons) qui est prévisible à partir des variables indépendantes (les caractéristiques des maisons).
Une valeur de R² de 0.9179 signifie que 91.79 % de la variance des prix des maisons peut être expliquée par le modèle.
Cela indique une très bonne performance du modèle. 
En général :
Un R² proche de 1 suggère que le modèle explique presque toute la variance des données.
Un R² proche de 0 indiquerait que le modèle n'explique pratiquement aucune variance.
### Voyant le R² ajusté
# Calculer R² ajusté
n = len(y_test)  # Nombre d'observations
p = X_test.shape[1]  # Nombre de prédicteurs
r2 = metrics.r2_score(y_test, y_predict)
r2_adjusted = 1 - (1 - r2) * (n - 1) / (n - p - 1)
print("R² ajusté:", r2_adjusted)
##### Ces résultats montrent un modèle robuste et bien ajusté, suggérant que les variables choisies capturent bien les variations de la variable cible.
##### Le fait qu'il soit très proche de R² (0.9144 contre 0.9147) indique que les variables incluses dans le modèle sont probablement pertinentes et qu’il n’y a pas d'overfitting important.
## **Classification avec Matrice de Confusion**
##### Préparer les données
data_cleaned_imputed['Price_Category'] = pd.qcut(data_cleaned_imputed['Price'], q=3, labels=['Low', 'Medium', 'High'])

# Préparation des données pour la matrice de confusion
y_true_categories = data_cleaned_imputed.loc[X_test.index, 'Price_Category']  # Catégories réelles
y_pred_categories = pd.qcut(y_predict, q=3, labels=['Low', 'Medium', 'High'])  # Catégories prédites

# Assuming y_true_categories and y_pred_categories are defined correctly
cm = confusion_matrix(y_true_categories, y_pred_categories, labels=['Low', 'Medium', 'High'])

##### Affichage de la matrice de confusion
# Affichage de la matrice de confusion
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low', 'Medium', 'High'])
disp.plot(cmap='Blues')
plt.title('Matrice de Confusion')
plt.show()
#### Ordinary Least Squares
# Implémentation de la régression OLS
# Ajouter une constante pour l'intercept
X_ols = sm.add_constant(X)  # X avec une constante
model_ols = sm.OLS(y, X_ols).fit()  # Ajuster le modèle OLS
print(model_ols.summary())  # Afficher le résumé du modèle
## Application de Adaptive Decision Tree ADD
# Prétraitement des Données
# Vérifier les colonnes et types de données
print("Colonnes originales :")
print(data_cleaned_imputed.columns)
print("\nTypes de données :")
print(data_cleaned_imputed.dtypes)
# Renommer les colonnes pour éviter les espaces
data_cleaned_imputed.columns = [col.replace(' ', '_') for col in data_cleaned_imputed.columns]
# Supprimer les colonnes non numériques
data_cleaned_numeric = data_cleaned_imputed.select_dtypes(include=['int64', 'float64'])
print("\nColonnes numériques :")
print(data_cleaned_numeric.columns)
# Définir les variables indépendantes (X) et la variable dépendante (y)
X = data_cleaned_numeric.drop(columns=['Price'])  # Toutes les colonnes sauf 'Price'
y = data_cleaned_numeric['Price']  # La colonne 'Price'

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # test_size est maintenant correctement spécifié
# Créer et entraîner le modèle de régression
regressor = DecisionTreeRegressor(random_state=42)
regressor.fit(X_train, y_train)
# Faire des prédictions et évaluer le modèle
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
# Calculer le R² ajusté
n = X_test.shape[0]  # Nombre d'observations
p = X_test.shape[1]  # Nombre de variables indépendantes
r2_adjusted = 1 - (1 - r2) * (n - 1) / (n - p - 1)
# Afficher les résultats
print(f'\nMean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')
print(f'Adjusted R-squared: {r2_adjusted:.2f}')
#### Une précision de 74% de notre modèle d'Arbre de Décision (ADD) dans la prédiction des prédictions des prix des maisons est un résultat acceptable cela signifie que les prédictions faites par le modèle sont correctes par rapport à l'ensemble de test. 
#### Diagramme de Dispersion
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Valeurs Réelles')
plt.ylabel('Valeurs Prédites')
plt.title('Valeurs Réelles vs Valeurs Prédites')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # ligne d'égalité
plt.show()
#### Histogramme des Résidus
residuals = y_test - y_pred
plt.hist(residuals, bins=30)
plt.xlabel('Résidus')
plt.ylabel('Fréquence')
plt.title('Histogramme des Résidus')
plt.show()
#### Graphique des Erreurs
plt.scatter(y_test, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Valeurs Réelles')
plt.ylabel('Résidus')
plt.title('Résidus vs Valeurs Réelles')
plt.show()
### Exactitude (Accuracy) : 0.021
L'exactitude de 0.021 indique que le modèle a correctement prédit environ 2,1 % des cas.
### Moyenne Macro (Macro Avg) :
Précision : 0.00 (Précision de chaque classe considérée individuellement, sans tenir compte du déséquilibre des classes)
Rappel : 0.01 (Proportion de vrais positifs sur le total des vrais positifs + faux négatifs)
F1-Score : 0.00 (Moyenne harmonique entre précision et rappel)
### Moyenne Pondérée (Weighted Avg) :
Précision : 0.00, Rappel : 0.02, F1-Score : 0.00
# Modèle de Classification par Etat
import re
# Function to extract state abbreviation from the Address column
def extract_state(address):
    # Regular expression to match two uppercase letters, optionally followed by a space and zip code
    match = re.search(r'\b[A-Z]{2}\b', str(address))
    return match.group(0) if match else None

# Apply the function to create the 'State' column
data_cleaned_imputed['State'] = data_cleaned_imputed['Address'].apply(extract_state)

state_names = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "AS": "American Samoa",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "MH": "Marshall Islands",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "PW": "Palau",
    "MP": "Northern Mariana Islands",
    "VI": "Virgin Islands",
    "PR": "Puerto Rico",
    "GU": "Guam",
    "FM": "Federated States of Micronesia",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia", "AP": "Armed Forces Pacific",
    "AE": "Armed Forces Europe", "AA": "Armed Forces Americas"
}

# Map state abbreviations to full names in a new column 'State Full Name'
data_cleaned_imputed['State Full Name'] = data_cleaned_imputed['State'].map(state_names)

output_path = 'USA_Housing_with_State_v3.csv'  # Choose a unique name for the new file
data.to_csv(output_path, index=False)
data = data_cleaned_imputed

output_path
# Définir X et y pour la classification
X = data.drop(columns=['State', 'State Full Name', 'Address', 'Price'])  # enlevez les colonnes non nécessaires
y = data['State']  # variable cible
# Séparer les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Sur-échantillonnage (si déséquilibre)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print (y_train_res)
# Initialiser le modèle Random Forest
model = RandomForestClassifier(random_state=42)
# Entraîner le modèle
model.fit(X_train_res, y_train_res)
# Faire des prédictions
y_pred = model.predict(X_test)
# Évaluer le modèle
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
#### Performance très faible représenter par une prédiction très faible 1.1%.
### Quelque métriques:
# Remplacez par votre modèle
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
y_pred = classifier.predict(X_test)

# Affichage des métriques
print("Métriques de performance du modèle de classification :")
print(f'Exactitude : {accuracy_score(y_test, y_pred)}')

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
print("Matrice de confusion :")
print(cm)

# Affichage de la matrice de confusion
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Affichage du rapport de classification
print(classification_report(y_test, y_pred, zero_division=0))

# Vérifiez les classes dans y_test
unique_classes = np.unique(y_test)
print("Classes dans y_test :", unique_classes)

if len(unique_classes) < 2:
    print("AUC ne peut pas être calculé car il n'y a qu'une seule classe dans y_test.")
else:
    # Binarisation des labels pour AUC
    y_test_binarized = label_binarize(y_test, classes=unique_classes)

    # Prédictions de probabilité
    y_proba = classifier.predict_proba(X_test)

    # Calculer le score AUC pour chaque classe
    roc_auc = roc_auc_score(y_test_binarized, y_proba, average='macro', multi_class='ovr')
    print(f'AUC: {roc_auc}')
#### Exactitude (Accuracy) :
L'exactitude est de 0.02, ce qui signifie que le modèle a correctement prédit seulement 2% des échantillons testés. 
#### Moyenne Macro : 
Les valeurs de 0.02 pour la précision, le rappel et la F1-mesure indiquent que le modèle a une performance très faible sur toutes les classes.
#### Moyenne Pondérée :  
Les mêmes valeurs que la moyenne macro sont également observées ici, ce qui est inhabituel.
#### AUC (Area Under the Curve) :
La valeur AUC de 0.494 est proche de 0.5, ce qui indique que le modèle ne parvient pas à distinguer les classes de manière significative
# Modèle de Classification par Etat SVM : Support Vector Machine
# Définir X et y pour la classification
X = data_cleaned_imputed.drop(columns=['State', 'State Full Name', 'Address', 'Price'])  # enlevez les colonnes non nécessaires
y = data_cleaned_imputed['State']  # variable cible
# Appliquer SMOTE pour équilibrer les classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
# Vérifiez les formes et les valeurs nulles
print(f'X_resampled shape: {X_resampled.shape}')
print(f'y_resampled shape: {y_resampled.shape}')
print(X_resampled.isnull().sum())
print(y_resampled.isnull().sum())
# Test de séparation sans le reste du code
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
print(f'X_train shape: {X_train.shape}, X_test shape: {X_test.shape}')

# Entraîner le modèle SVM

svm_model = SVC(kernel='linear', random_state=42) # Vous pouvez changer le kernel si nécessaire
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
svm_model.fit(X_train, y_train)

# Faire des prédictions
y_pred = svm_model.predict(X_test)
# Évaluer le modèle
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.3f}')
# print("Classification Report:")
# print(classification_report(y_test, y_pred))
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, zero_division=1))
# Calculer la moyenne des prix par état
average_price_per_state = data_cleaned_imputed.groupby('State Full Name')['Price'].mean().reset_index()

# Renommer la colonne pour plus de clarté
average_price_per_state.columns = ['State', 'Average Price']

# Afficher le résultat
print(average_price_per_state)

# Calculer la moyenne des colonnes spécifiées par état
average_data_per_state = data_cleaned_imputed.groupby('State Full Name').agg({
    'Avg. Area Income': 'mean',
    'Avg. Area House Age': 'mean',
    'Avg. Area Number of Rooms': 'mean',
    'Avg. Area Number of Bedrooms': 'mean',
    'Area Population': 'mean',
    'Price': 'mean'
}).reset_index()

# Renommer les colonnes pour plus de clarté
average_data_per_state.columns = ['State', 'Average Area Income', 
                                   'Average Area House Age', 
                                   'Average Number of Rooms', 
                                   'Average Number of Bedrooms', 
                                   'Average Population', 
                                   'Average Price']

# Afficher le résultat
print(average_data_per_state)

# Enregistrer la nouvelle base de données dans un fichier CSV
output_path = 'Average_Data_Per_State.csv'
average_data_per_state.to_csv(output_path, index=False)
# Définir X et y pour la classification
X = data_cleaned_imputed.drop(columns=['State', 'State Full Name', 'Address', 'Price'])  # enlevez les colonnes non nécessaires
y = data_cleaned_imputed['State']  # variable cible
# Appliquer SMOTE pour équilibrer les classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Vérifiez les formes et les valeurs nulles
print(f'X_resampled shape: {X_resampled.shape}')
print(f'y_resampled shape: {y_resampled.shape}')
print(X_resampled.isnull().sum())
print(y_resampled.isnull().sum())
# Test de séparation sans le reste du code
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
print(f'X_train shape: {X_train.shape}, X_test shape: {X_test.shape}')

# Entraîner le modèle SVM

svm_model = SVC(kernel='linear', random_state=42) # Vous pouvez changer le kernel si nécessaire
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
svm_model.fit(X_train, y_train)
# Faire des prédictions
y_pred = svm_model.predict(X_test)
# Évaluer le modèle
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.3f}')
#### Une précision de 2,5 % est très faible et indique que votre modèle ne fonctionne pas bien
### Quelque métriques:
# Supposons que vous avez déjà vos données chargées dans X et y
# X = vos données d'entrée
# y = vos labels

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Création et entraînement du modèle SVM
svm_model = svm.SVC(probability=True)  # Assurez-vous que probability=True
svm_model.fit(X_train, y_train)

# Prédictions sur l'ensemble de test
y_pred = svm_model.predict(X_test)

# Affichage du rapport de classification
print(classification_report(y_test, y_pred, zero_division=0))

# Vérifiez les classes dans y_test
unique_classes = np.unique(y_test)
print("Classes dans y_test :", unique_classes)

if len(unique_classes) < 2:
    print("AUC ne peut pas être calculé car il n'y a qu'une seule classe dans y_test.")
else:
    # Binarisation des labels pour AUC
    y_test_binarized = label_binarize(y_test, classes=unique_classes)

    # Prédictions de probabilité
    y_proba = svm_model.predict_proba(X_test)

    # Calculer le score AUC pour chaque classe
    roc_auc = roc_auc_score(y_test_binarized, y_proba, average='macro', multi_class='ovr')
    print(f'AUC: {roc_auc}')

# Calcul de la matrice de confusion
cm = confusion_matrix(y_test, y_pred, labels=unique_classes)

# Affichage de la matrice de confusion
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_classes)
disp.plot(cmap=plt.cm.Blues)  # Utilisez un colormap pour une meilleure visibilité
plt.title('Matrice de Confusion')
plt.show()
#### Exactitude (Accuracy) :
0.02 : Cela signifie que le modèle a correctement prédit 2 % des échantillons testés.
#### Moyenne Macro : 
Les valeurs pour la précision, le rappel et la F1-mesure sont très faibles, avec un score de 0.00 pour la précision et la F1-mesure, et 0.01 pour le rappel. 
#### Moyenne Pondérée : 
Les résultats ici montrent également des valeurs proches de zéro.
#### AUC (Area Under the Curve) :
L'AUC de 0.511 est légèrement au-dessus de 0.5, ce qui indique que le modèle a une capacité de discrimination très limitée, proche du tirage aléatoire.
