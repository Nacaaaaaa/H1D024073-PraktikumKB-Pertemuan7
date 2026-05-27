# Dokumentasi Praktikum Jaringan Syaraf Tiruan - Klasifikasi Iris

**Identitas Mahasiswa:**
* Nama: Nalendra Wicaksana
* NIM: H1D024073

### 1. Tujuan Penugasan
Mengembangkan sistem klasifikasi berbasis Jaringan Syaraf Tiruan (JST) Multi-Layer Perceptron untuk memprediksi spesies bunga Iris berdasarkan matriks dimensi kelopak (sepal) dan mahkota (petal) secara lokal menggunakan framework TensorFlow dan Keras.

### 2. Struktur Repositori
- main_iris.py : Berisi baris kode pemrograman Python utama.
- iris.data : Dataset numerik kelopak dan mahkota bunga Iris.
- README.md : Berisi panduan teknis dan analisis operasional sistem.

### 3. Penjelasan Kerja Kode dan Dampak Output

#### Petikan Kode Bagian 1:
dataset = pd.read_csv('iris.data', header=None, sep=',')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
Penjelasan Operasional:
Baris ini menginstruksikan modul Pandas untuk membaca file eksternal lokal bernama 'iris.data'. Objek dipecah ke dalam array dua dimensi melalui fungsi 'iloc'. Variabel 'X' mengambil seluruh kolom kecuali kolom terakhir sebagai representasi matriks dimensi fisik bunga, sedangkan variabel 'y' mengisolasi kolom akhir sebagai target target spesies.
Dampak Output:
Menghasilkan struktur data matriks numerik berukuran 150 sampel dengan 4 parameter input, serta vektor label bertipe string sepanjang 150 entri.

Petikan Kode Bagian 2:
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
Penjelasan Operasional:
Fungsi 'LabelEncoder' dari pustaka scikit-learn melakukan pemetaan data kategori string pada variabel 'y' menjadi bentuk indeks integer berurutan secara otomatis.
Dampak Output:
Entri string bertransformasi menjadi representasi numerik terstandarisasi, di mana kategori 'Iris-setosa' diwakili angka 0, 'Iris-versicolor' angka 1, dan 'Iris-virginica' angka 2.

Petikan Kode Bagian 3:
model = Sequential([
    Input(shape=X_train.shape[1:]),
    Dense(1000, activation='relu'),
    Dense(500, activation='relu'),
    Dense(300, activation='relu'),
    Dense(3, activation='softmax')
])
Penjelasan Operasional:
Mendefinisikan topologi jaringan saraf secara linier berurutan menggunakan model 'Sequential'. Lapisan 'Input' mendeteksi jumlah dimensi masukan. Tiga lapisan tersembunyi ('Dense') dikonfigurasi masing-masing dengan alokasi 1000, 500, dan 300 unit neuron yang dilengkapi fungsi non-linieritas 'ReLU'. Lapisan keluaran akhir berisi 3 unit neuron menggunakan fungsi aktivasi 'Softmax'.
Dampak Output:
Terbentuk struktur model komputasi berlapis. Penggunaan fungsi 'Softmax' pada ujung lapisan memastikan bahwa akumulasi total nilai keluaran dari ketiga neuron menghasilkan probabilitas pecahan berbobot total 1.0.

Petikan Kode Bagian 4:
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
Penjelasan Operasional:
Fungsi 'compile' mengonfigurasi parameter pembelajaran model dengan menetapkan algoritma optimasi 'Adam' dan fungsi kerugian kerangka kerja 'sparse_categorical_crossentropy'. Proses dilanjutkan dengan fungsi 'fit' untuk mengeksekusi siklus pelatihan interaktif sebanyak 50 kali pengulangan (epochs) dengan kapasitas pemrosesan data per kelompok sebesar 32 baris (batch_size).
Dampak Output:
Terminal akan menampilkan visualisasi log pergerakan nilai kegagalan (loss) yang terus menurun dan tingkat akurasi (accuracy) yang meningkat di setiap tahapan epoch, baik pada himpunan data pelatihan maupun validasi.

Petikan Kode Bagian 5:
predictions = model.predict(X_test)
predicted_classes = predictions.argmax(axis=1)
Penjelasan Operasional:
Fungsi 'predict' memproses data uji untuk menghitung prediksi probabilitas. Perintah 'argmax(axis=1)' menyaring indeks neuron mana yang menghasilkan nilai persentase keyakinan paling tinggi dari matriks output.
Dampak Output:
Menghasilkan output berupa array angka integer (0, 1, atau 2) yang merepresentasikan keputusan mutlak hasil tebakan klasifikasi model AI terhadap subset data uji.

### 4. Panduan Eksekusi Repositori (Cloning)

Langkah 1: Mengunduh repositori kerja dari server GitHub ke komputer lokal.
Sintaks Perintah: git clone https://github.com/[Username-GitHub-Anda]/H1D024094-PraktikumKB-Pertemuan7.git

Langkah 2: Mengubah direktori aktif terminal menuju folder proyek hasil proses kloning.
Sintaks Perintah: cd H1D024094-PraktikumKB-Pertemuan7

Langkah 3: Memasang paket ketergantungan modul pustaka Python yang diwajibkan oleh sistem.
Sintaks Perintah: pip install tensorflow scikit-learn pandas numpy matplotlib seaborn

Langkah 4: Menjalankan skrip utama sistem klasifikasi kecerdasan buatan.
Sintaks Perintah: python main.py
