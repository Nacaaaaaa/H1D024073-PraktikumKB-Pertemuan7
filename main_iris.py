import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Input
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 1. Muat dataset iris dari file lokal [cite: 90-91]
dataset = pd.read_csv('iris.data', header=None, sep=',')

# 2. Menyusun data X (fitur) dan y (label) [cite: 92-93]
X = dataset.iloc[:, :-1].values 
y = dataset.iloc[:, -1].values  

# 3. Mengonversi label dari string menjadi numerik [cite: 96-98]
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y) 

# 4. Memisahkan dataset menjadi data latih dan data validasi [cite: 100-101]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Buat model neural network [cite: 104-107]
model = Sequential([
    Input(shape=X_train.shape[1:]),
    Dense(1000, activation='relu'),
    Dense(500, activation='relu'),
    Dense(300, activation='relu'),
    Dense(3, activation='softmax')
])

# 6. Melihat summary arsitektur model [cite: 110-111]
model.summary()

# 7. Kompilasi model [cite: 112-115]
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 8. Melatih model [cite: 118-124]
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# 9. Evaluasi model [cite: 129-131]
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Loss: {loss}, Accuracy: {accuracy}")

# Visualisasi loss dan accuracy [cite: 133-134]
pd.DataFrame(history.history).plot(figsize=(10,6))
plt.title('Training and Validation Metrics')
plt.show()

# 10. Prediksi pada data validasi [cite: 135-139]
predictions = model.predict(X_test)
predicted_classes = predictions.argmax(axis=1)
print("Prediksi:", predicted_classes)
print("Label Asli:", y_test)

# 11. Visualisasi Confusion Matrix [cite: 143-154]
cm = confusion_matrix(y_test, predicted_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# 12. Fungsi prediksi data input baru [cite: 155-168]
def predict_new_data():
    sepal_length = float(input("Masukkan sepal length: "))
    sepal_width = float(input("Masukkan sepal width: "))
    petal_length = float(input("Masukkan petal length: "))
    petal_width = float(input("Masukkan petal width: "))
    
    new_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(new_data)
    predicted_class = prediction.argmax(axis=1)
    predicted_label = label_encoder.inverse_transform(predicted_class)
    
    print(f"Prediksi kelas: {predicted_label[0]}")

predict_new_data()