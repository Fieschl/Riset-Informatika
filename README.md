# Analisis Kinerja Algoritma Convolutional Neural Network (CNN) Vanilla dalam Deteksi Penyakit pada Daun Kacang
Nama : Alif Maulana\
NPM : 20081010181\
Kelas : Riset Informatika D
## Problem Statement
Tanaman kacang merupakan salah satu sumber penting dalam produksi pangan global. Namun, serangan penyakit pada daun kacang dapat secara signifikan mengurangi hasil dan kualitas panen. Identifikasi manual penyakit pada daun kacang memerlukan waktu dan keahlian yang cukup, sehingga dapat menyebabkan keterlambatan dalam respons penanganan yang diperlukan.

Meskipun telah ada upaya menggunakan teknologi dalam mendeteksi penyakit pada tanaman, penggunaan algoritma Convolutional Neural Network (CNN) Vanilla masih memerlukan analisis lebih lanjut. Masih terdapat kebutuhan akan penelitian yang mengkaji secara mendalam performa serta kinerja algoritma CNN Vanilla dalam mendeteksi penyakit pada daun kacang secara akurat dan efisien.

Model CNN Vanilla diperlukan optimasi agar dapat mengenali dan mengklasifikasikan berbagai jenis penyakit yang umum terjadi pada daun kacang dengan tingkat akurasi yang tinggi. Selain itu, perlu juga dilakukan evaluasi terhadap faktor-faktor yang mempengaruhi kinerja model, seperti ukuran dataset, arsitektur jaringan, serta metode pre-processing yang digunakan.

Penelitian ini diharapkan dapat memberikan kontribusi signifikan dalam pengembangan sistem deteksi dini penyakit pada tanaman kacang melalui penerapan teknologi AI yang lebih efektif dan efisien.
## Research Question
* Bagaimana cara mengimplementasikan algoritma CNN dengan metode Vanilla dalam mendeteksi penyakit pada daun kacang?

* Seberapa akurat algoritma CNN dengan metode Vanilla dalam mendeteksi penyakit pada daun kacang?
## Dataset
Penelitian dilakukan menggunakan dataset yang terdiri dari total 990 sampel yang dibagi ke dalam tiga kategori utama. Sebanyak 330 sampel merupakan contoh dari daun kacang yang terkena bercak daun bersudut, 330 sampel lainnya mewakili kondisi daun kacang yang terinfeksi karat kacang, dan 330 sampel sisanya menggambarkan daun kacang dalam keadaan sehat. Dengan menggunakan dataset yang tersegmentasi secara proporsional, diharapkan penelitian dapat memberikan pemahaman yang lebih mendalam terkait kinerja algoritma dalam deteksi penyakit pada tanaman kacang dengan tingkat akurasi yang memadai.

## Contoh Dataset Yang Digunakan
![contoh-dataset](https://github.com/Fieschl/Riset-Informatika/assets/88276269/eaa26f9c-0b8f-4110-836f-1d824e899576)

## Distribusi Pada Setiap Sampel Daun
![diagram](https://github.com/Fieschl/Riset-Informatika/assets/88276269/7ed4679a-1dce-477c-9b3d-8b10b9f00d34)

Distribusi menunjukan bahwa setiap sampel memiliki jumlah yang seimbang. Dapat dilihat pada tabel diatas menunjukan bahwa persebaran data memiliki tingkat kesetaraan yang relatif sama yaitu 350.

## Data Training Yang Digunakan
![dataset-training](https://github.com/Fieschl/Riset-Informatika/assets/88276269/3a15f310-b2a2-40ae-ba27-170ed10759ab)
Dari 990 sampel yang dibagi ke dalam tiga kelas terdapat sebanyak 792 sampel digunakan untuk proses pelatihan model. Data yang tersisa sebanyak 198 sampel akan digunakan digunakan untuk validasi.

## Preprocessing
Preprocessing merupakan proses untuk membersihkan dan menyusun data agar mudah dimengerti oleh komputer. Prosesnya bisa mencakup menghapus data yang hilang, mengubah teks menjadi angka, dan memastikan semua data memiliki skala yang sama pentingnya. 

## Importing Dataset
```ruby
data_path = "/content/Bean_Dataset"
class_names = ["angular_leaf_spot", "bean_rust", "healthy"]
```

## Pengolahan Dataset
```ruby
BATCH_SIZE = None
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
SEED = 42

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_path,
    batch_size=BATCH_SIZE,
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    shuffle=True,
    seed=SEED,
    validation_split=0.2,
    subset="training",
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_path,
    batch_size=BATCH_SIZE,
    image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    shuffle=True,
    seed=SEED,
    validation_split=0.2,
    subset="validation",
)
```
Proses ini bertujuan untuk memuat dataset gambar dari direktori tertentu menggunakan TensorFlow. Dengan masing-masing data disesuaikan dengan parameter-parameter seperti ukuran batch, dimensi gambar, pengacakan data, dan pembagian dataset menjadi bagian pelatihan (80%) dan validasi (20%). Langkah ini membantu untuk memeriksa dan memastikan kecocokan serta kelengkapan arsitektur model sebelum proses pelatihan atau evaluasi dilakukan terhadap dataset yang diberikan.

## Permodelan CNN
```ruby
def conv_block(x, n_filters):
    x = tf.keras.layers.Conv2D(n_filters, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    return x

def dense_block(x, n_units, dropout_rate=0.2):
    x = tf.keras.layers.Dense(n_units, activation="relu", kernel_regularizer='l2')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x

def create_cnn(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Rescaling(1./255)(inputs)

    # conv block 1
    conv1 = conv_block(x, 64)
    # conv block 2
    conv2 = conv_block(conv1, 128)
    # conv block 3
    conv3 = conv_block(conv2, 256)
    # conv block 4
    conv4 = conv_block(conv3, 512)
    # flatten
    flat = tf.keras.layers.GlobalAveragePooling2D()(conv4)
    # classification head
    dense1 = dense_block(flat, 256)
    dense2 = dense_block(dense1, 25)
    outputs = tf.keras.layers.Dense(3, activation="softmax")(dense2)
    
    model = tf.keras.Model(inputs, outputs)
    return model
```
Permodelan Convolutional Neural Network (CNN) dengan TensorFlow yang terdiri dari blok-blok konvolusi dan lapisan-lapisan neural network. Dimulai dari proses ekstraksi fitur melalui konvolusi hingga pembentukan output klasifikasi. Permodelan ini digunakan untuk tugas klasifikasi pada dataset gambar setelah dilakukan pelatihan menggunakan data yang sesuai.

![permodelan](https://github.com/Fieschl/Riset-Informatika/assets/88276269/ac7de83f-1c66-43ac-9983-b1356368b026)

## Implementasi Convolutional Neural Network (CNN) Vanilla
Setelah proses preprocessing, data yang telah diolah akan disajikan ke dalam model machine learning untuk dilakukan pelatihan. Data yang telah dilatih akan digunakan untuk menentukan ketepatan akurasi dari algoritma Convolutional Neural Network (CNN) Vanilla dalam melakukan deteksi penyakit pada daun tanaman kacang.

## Konfigurasi Convolutional Neural Network (CNN) Vanilla
```ruby
cnn.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15),
    tf.keras.callbacks.ModelCheckpoint("cnn", save_best_only=True),
]

epochs = 10
```

Dilakukan pengaturan awal, kompilasi, pelatihan, dan peningkatan performa model CNN untuk tugas klasifikasi. Terdapat beberapa poin penting dari tahapan ini:

1.	Compile Model: Menentukan konfigurasi optimizer, fungsi loss, dan metrik evaluasi untuk model.
2.	Callbacks: Menggunakan callbacks seperti EarlyStopping untuk menghentikan pelatihan saat kriteria tertentu tidak terpenuhi dan ModelCheckpoint untuk menyimpan model terbaik selama pelatihan.
3.	Proses Pelatihan: Menentukan jumlah iterasi (epoch) yang akan dilakukan saat melatih model. Dalam hal ini akan dilakukan iterasi sebanyak 10 kali.

## Training Dataset Dengan Convolutional Neural Network (CNN) Vanilla
```ruby
cnn_history = cnn.fit(
    train_batches,
    validation_data=val_batches,
    epochs=epochs,
    callbacks=callbacks,
)
```
Tahapan ini merupakan proses pelatihan (training) untuk model CNN (Convolutional Neural Network) dengan menggunakan TensorFlow dan Keras.

## Hasil Training Dataset
![epoch-nwe](https://github.com/Fieschl/Riset-Informatika/assets/88276269/f3d6e34e-a6a3-4f43-b971-cb92e8383bce)

# Performa Neural Network (CNN) Vanilla
![training-diagram](https://github.com/Fieschl/Riset-Informatika/assets/88276269/a5dd4496-4b16-4e97-986a-6b90b973422b)

Pengujian yang dilakukan dengan menggunakan nilai learning rate 10 epoch pada data latih dan data validasi. Pada proses pelatihan model menghasilkan nilai accuracy dan nilai loss yang dapat dilihat pada kedua gambar diatas.

## Analisis Pengujian Convolutional Neural Network (CNN) Vanilla
Pada tahapan ini dilakukan analisis untuk menilai sejauh mana akurasi model Convolutional Neural Network (CNN) Vanilla dalam melakukan klasifikasi penyakit pada daun kacang. Pengujian metode akan dilakukan dengan menggunakan Confusion Matriks.

## Confusion Matriks
![matrix](https://github.com/Fieschl/Riset-Informatika/assets/88276269/b7828bfa-0664-4832-9952-d329ab28c834)

Confusion matrix merupakan representasi dari kinerja model klasifikasi yang menunjukkan seberapa baik atau buruk model tersebut dalam melakukan klasifikasi pada setiap kelas. 

## Classification Report
![report-calisifaction](https://github.com/Fieschl/Riset-Informatika/assets/88276269/114fd33e-1627-4990-aa44-b2396af6c487)

daun kacang yang terkena bercak daun bersudut, 330 sampel lainnya mewakili kondisi daun kacang yang terinfeksi karat kacang, dan 330 sampel sisanya menggambarkan daun kacang dalam keadaan sehat.

Dari data laporan klasifikasi diatas ditemukan bahwa:
* Pada kelas daun kacang yang terkena bercak daun bersudut (angular_leaf_spot) terdapat 28 sampel yang sesuai dalam kelas tersebut. Model berhasil memprediksi dengan benar sebanyak 0.84 atau 84% dari keseluruhan contoh dalam kelas tersebut. Recall (sensitivitas) yang ditunjukkan adalah 0.57 atau 57%, yang berarti 57% dari sampel yang sesuai dalam kelas tersebut berhasil diprediksi dengan benar oleh model.

* Pada kelas daun kacang yang terinfeksi karat kacang (bean rust) terdapat 42 sampel yang sesuai dalam kelas tersebut. Model berhasil memprediksi dengan benar sebanyak 0.80 atau 80% dari keseluruhan sampel yang terdapat dalam kelas tersebut. Dengan recall dari kelas "bean_rust" adalah 0.79 atau 79%.

* Pada kelas daun yang tergolong sehat (healthy) terdapat 30 sampel yang sesuai dalam kelas tersebut. Model berhasil memprediksi dengan benar sebanyak 0.75 atau 75% dari keseluruhan sampel. Recall kelas "healthy" adalah 1.00 atau 100%, yang menunjukkan bahwa model dapat memprediksi seluruh sampel dengan benar pada kelas tersebut.

## Kesimpulan
Algoritma Convolutional Neural Network (CNN) Vanilla dalam melakukan klasifikasi penyakit pada daun kacang menunjukkan akurasi yang tinggi. Secara keseluruhan model dapat melakukan prediksi secara akurat dengan nilai akurasi total model adalah 0.79 atau 79%. Model ini berhasil mengenali penyakit yang terdapat pada daun tanaman kacang. Akan tetapi, dari hasil akhir akurasi model masih diperlukan ruang untuk dilakukan evaluasi perbaikan. Penyempurnaan model diperlukan agar dapat bekerja lebih optimal dalam mengenali jenis penyakit yang terdapat dalam tanaman kacang.

## Literatur Review
* Aziz, Musthafa, Syaikhul Anam Alidrus, and Oddy Virgantara Putra. (2021). Deteksi Penyakit Pada Daun Tanaman Padi Menggunakan Metode Convolutional Neural Network. SENAMIKA 103-109.
* Aishwarya, M. P., & Reddy, P. (2023). Ensemble of CNN models for classification of groundnut plant leaf. Smart Agricultural Technology.



