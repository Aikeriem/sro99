import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
Q8 = pd.read_excel('c:\\Users\\User\\Desktop\\pyt\\Dataset_pandas_assign.xlsx')
Q8 = Q8.groupby('Город', as_index=False)['Дата проишествия'].count().sort_values('Дата проишествия', ascending=False).head(10)
Q8.rename(columns={'Дата проишествия': 'Кол-во дорожных происшествий'}, inplace=True)

cities = ['город1', 'город2'] 
Q8['Местность'] = Q8['Город'].apply(lambda x: 'Город' if x in cities else 'Сельская местность')
X = Q8[['Кол-во дорожных происшествий']]
y = Q8['Местность']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = lgb.LGBMClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели: {accuracy}")
insurance = pd.read_excel('c:\\Users\\User\\Desktop\\pyt\\Dataset_pandas_assign.xlsx')
gender_mapping = {'мужской': 0, 'женский': 1}
insurance['Пол'] = insurance['Пол'].map(gender_mapping)
X = insurance[['Пол', 'Стаж вождения']]
y = insurance['КБМ']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Среднеквадратичная ошибка: {mse}")
