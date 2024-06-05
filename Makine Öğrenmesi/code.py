import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import requests

# Veri setini indir
url = "https://s3.cloud.ngn.com.tr/clu3-40/course/461189/activity/479719/veri-seti.txt?AWSAccessKeyId=ALMS%3aalms-storage%40advancity.com.tr&Expires=1717594234&Signature=papy9dxPWmTeAVJ8OYQr0j5MBcg%3d"
response = requests.get(url)
with open('veri-seti.txt', 'wb') as file:
    file.write(response.content)

# Veri setini yükle ve sütunları ayır
column_names = ['Hamilelik', 'Glikoz', 'KanBasinci', 'CiltKalınligi', 'Insulin', 'BMI', 'DiabetesPedigreeFonksiyonu', 'Yas', 'Sonuc']
df = pd.read_csv('veri-seti.txt', delimiter='\t', header=None, names=column_names)

# Eksik değerlerin kontrolü ve doldurulması
df[['Glikoz', 'KanBasinci', 'CiltKalınligi', 'Insulin', 'BMI']] = df[['Glikoz', 'KanBasinci', 'CiltKalınligi', 'Insulin', 'BMI']].replace(0, np.nan)
df.fillna(df.mean(), inplace=True)

# Veri setini özellikler (X) ve hedef değişken (y) olarak ayır
X = df.drop('Sonuc', axis=1)
y = df['Sonuc']

# Veriyi standartlaştır
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Veri setini eğitim ve test olarak ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Naive Bayes modeli eğit ve tahmin yap
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

# Sonuçları raporla
print("Naive Bayes Sınıflandırıcısı")
print(confusion_matrix(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))

# ROC eğrisi çizimi için gerekli skorlar
y_prob_nb = nb_model.predict_proba(X_test)[:, 1]
fpr_nb, tpr_nb, _ = roc_curve(y_test, y_prob_nb)
roc_auc_nb = auc(fpr_nb, tpr_nb)

# K-NN modeli ve GridSearchCV ile en iyi k değerini bulma
param_grid = {'n_neighbors': np.arange(1, 50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, param_grid, cv=5)
knn_cv.fit(X_train, y_train)

# En iyi k değeri ile model eğitimi ve tahmin yapma
best_k = knn_cv.best_params_['n_neighbors']
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

# Sonuçları raporla
print(f"K-En Yakın Komşuluk Sınıflandırıcısı (En iyi k={best_k})")
print(confusion_matrix(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

# ROC eğrisi çizimi için gerekli skorlar
y_prob_knn = knn.predict_proba(X_test)[:, 1]
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_prob_knn)
roc_auc_knn = auc(fpr_knn, tpr_knn)

# MLP modeli eğitimi ve tahmin yapma
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)

# Sonuçları raporla
print("Çok Katmanlı Algılayıcı (MLP) Sınıflandırıcısı")
print(confusion_matrix(y_test, y_pred_mlp))
print(classification_report(y_test, y_pred_mlp))

# ROC eğrisi çizimi için gerekli skorlar
y_prob_mlp = mlp.predict_proba(X_test)[:, 1]
fpr_mlp, tpr_mlp, _ = roc_curve(y_test, y_prob_mlp)
roc_auc_mlp = auc(fpr_mlp, tpr_mlp)

# SVM modeli eğitimi ve tahmin yapma
svm = SVC(probability=True, random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

# Sonuçları raporla
print("Destek Vektör Makineleri (SVM) Sınıflandırıcısı")
print(confusion_matrix(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

# ROC eğrisi çizimi için gerekli skorlar
y_prob_svm = svm.predict_proba(X_test)[:, 1]
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_prob_svm)
roc_auc_svm = auc(fpr_svm, tpr_svm)

# ROC eğrisi çizimi
plt.figure(figsize=(10, 8))
plt.plot(fpr_nb, tpr_nb, label=f'Naive Bayes (AUC = {roc_auc_nb:.2f})')
plt.plot(fpr_knn, tpr_knn, label=f'K-NN (AUC = {roc_auc_knn:.2f})')
plt.plot(fpr_mlp, tpr_mlp, label=f'MLP (AUC = {roc_auc_mlp:.2f})')
plt.plot(fpr_svm, tpr_svm, label=f'SVM (AUC = {roc_auc_svm:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Yanlış Pozitif Oranı')
plt.ylabel('Doğru Pozitif Oranı')
plt.title('ROC Eğrisi')
plt.legend(loc='lower right')
plt.show()
