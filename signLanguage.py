import multiprocessing
import numpy as np
import cv2
import tensorflow.keras as tf
import math
import os

#etiketleri labels.txt dosyasından alma
labels_path = "converted_keras/labels.txt"
labelsfile = open(labels_path, 'r')

#sınıflar için liste oluşturup onları sona kadar ekliyoruz
classes = []
line = labelsfile.readline()
while line:
        # sınıf isimlerini getirip sınıflar listemize ekliyoruz
        classes.append(line.split(' ', 1)[1].rstrip())
        line = labelsfile.readline()
labelsfile.close()

# teachable machine modelimizi yüklüyoruz
model_path = "converted_keras/keras_model.h5"
model = tf.models.load_model(model_path, compile=False)

cap = cv2.VideoCapture(0) #kamerayı aktif hale getirdik

# kamera genişliği ve yüksekliğini ayarlıyoruz
frameWidth = 1280
frameHeight = 720

# genişliği ve yükseliği pixel cinsinden ayarlıyoruz
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)
cap.set(cv2.CAP_PROP_GAIN, 0)

while True:
        # bilimsel gösterim doğruluk için devre dışı bırakıldı
        #np.set_printoptions(suppress=False)

        # Keras modeline yüklemek için doğru şeklin dizisini oluşturun.
        # Biz 1x 224x224 pixel RGB resim yüklüyoruz.
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        # Fotoğrafı çek
        check, frame = cap.read()
        frame = cv2.flip(frame, 1)  # flip komutu ile sağ ve sol yöndeki tersliği düzelt

        margin = int(((frameWidth-frameHeight)/10)) # web cam boyutunu ayarlıyoruz
        square_frame = frame[0:frameHeight, margin:margin + frameHeight]
        # TM model'de kullanmak üzere resmi 224x224 boyutlarında kırpıyoruz
        resized_img = cv2.resize(square_frame, (224, 224))
        # convert image color to go to model
        model_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

        # resim dizisini numpy dizisine çeviriyoruz
        image_array = np.asarray(model_img)
        # normalized ediyoruz
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        # resmi normalized diziye yolluyoruz
        data[0] = normalized_image_array

        # tahmin threshold'unu 90% olarak ayarlıyoruz.
        conf_threshold = 90
        confidence = []
        threshold_class = ""

        # tahmin etmeyi çalıştırıyoruz ve console'a yazdırmayı sağlıyoruz
        predictions = model.predict(data)
        if predictions[0,0] > conf_threshold / 100:
            print("Default")
        elif predictions[0,1] > conf_threshold / 100:
            print("Merhaba");
        elif predictions[0,2] > conf_threshold / 100:
            print("Nasilsin?");
        elif predictions[0,3] > conf_threshold / 100:
            print("Adin ne?");
        elif predictions[0,4] > conf_threshold / 100:
            print("Gule gule");
        elif predictions[0, 5] > conf_threshold / 100:
            print("Komik !");
        elif predictions[0,6] > conf_threshold / 100:
            print("Sag ol");

        # aşağıya siyah kenar ekler
        per_line = 2  # number of classes per line of text
        bordered_frame = cv2.copyMakeBorder(
            square_frame,
            top=0,
            bottom= 20 + 15*math.ceil(len(classes)/per_line),
            left= 0,
            right= 0,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )

        # her bir sınıf için
        for i in range(0, len(classes)):
            # tahmin güvenirliğini % olarak ölçüp listeye ekliyoruz
            confidence.append(int(predictions[0][i]*100))

            # oranı geçince durumu sıraya ekler
            if confidence[i] > conf_threshold:
                threshold_class = classes[i]

        # durumu siyah kısma yazdırır
        cv2.putText(
            img=bordered_frame,
            text=threshold_class,
            org=(10, frameHeight - 200),  # 720  - 525
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.9,
            color=(255, 255, 255),
        )

        # kamerayı göster
        cv2.imshow('WebCam', bordered_frame)
        cv2.waitKey(1)