#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow
import keras


# In[12]:


from os import listdir
from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model

from numpy import array
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint


# In[ ]:


### FOTOĞRAF VERİSİNİ HAZIRLA


# In[67]:


#bir dizin adı verildiğinde, her fotoğrafı yükleyen, VGG için hazırlayacak ve VGG modelinden tahmin edilen özellikleri toplayan fonksiyon. 
#Görüntü özellikleri, 1 boyutlu 4.096 elemanlı vektör.
#fonksiyon görüntü özelliklerine, görüntü tanımlayıcı bir dizi döndürüyor.
#klasördeki her fotoğrafın özelliklerini çıkartalım
def ozellik_ayikla(dizin):
    #modeli yükle
    model = VGG16()
    # modeli düzenle
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    # özet
    print(model.summary())
    # her fotoğrafın özelliklerini çıkaralım
    ozellikler = dict()
    for name in listdir(dizin):
        # dosyadan resim yükle
        dosyaismi = dizin + '/' + name
        image = load_img(dosyaismi, target_size=(224, 224))
        # pikselleri numpy arraya çevirelim
        image = img_to_array(image)
        # veriyi model için yeniden düzenle
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # VGG modeli için resmi hazırla
        image = preprocess_input(image)
        # özellikleri al
        feature = model.predict(image, verbose=0)
        # resim id
        foto_id = name.split('.')[0]
        # özelliği sakla
        ozellikler[foto_id] = feature
        print('>%s' % name)
    return ozellikler


# In[17]:


# tüm görüntülerin özelliklerini çıkar
#burada yukarıda tanımladığımız fonksiyonu kullanarak modellerimizi test etmek için fotoğraf verisini hazırladık.
#ardından ortaya çıkan diziyi 'features.pkl' olarak kaydettik.
dizin = 'Flicker8k_Dataset'
ozellikler = ozellik_ayikla(dizin)
print('Tüm Özellikler: %d' % len(ozellikler))
# dosyaya kaydet
dump(ozellikler, open('features.pkl', 'wb'))


# In[ ]:


### METİN VERİSİNİ HAZIRLA


# In[68]:


#veriseti her fotoğraf için birden fazla açıklama içeriyor ve burada küçük bir düzeyde temizlik yapmamız gerekiyor.
# dosyayı hafızaya yükle
def dosya_yukle(dosyaismi):
    # dosyayı salt okunur olarak aç
    dosya = open(dosyaismi, 'r')
    # bütün metni oku
    text = dosya.read()
    # dosyayı kapat
    dosya.close()
    return text

dosyaismi = 'sss-2.txt'
# açıklamaları yükle
doc = dosya_yukle(dosyaismi)


# In[69]:


#her fotoğrafın benzersiz bir tanımlayıcısı vardır. bu tanımlayıcı, fotoğraf dosyası adında ve açıklamaların metin dosyasında kullanılır.
#daha sonra fotoğraf açıklamaları listesine bakacağız. Aşağıda yüklenen belge metni verildiğinde açıklamalara bir fotoğraf tanımlayıcı dictionary döndürecek fonksiyonumuz var.
#her fotoğraf tanımlayıcısı, bir ve birden fazla metinsel açıklamanın listesiyle eşleşiyor.

# resimler için açıklamaları çıkar
def aciklamalari_yukle(doc):
    mapping = dict()
    # satırları işle
    for line in doc.split('\n'):
        # çizgiyi boşlukla böl
        tokens = line.split()
        if len(line) < 2:
            continue
        # başlangıç resim kimliği, gerisini açıklama olarak al
        foto_id, foto_aciklama = tokens[0], tokens[1:]
        # dosya adını resim kimliğinden kaldır
        foto_id = foto_id.split('.')[0]
        # açıklamaları metne döndür
        foto_aciklama = ' '.join(foto_aciklama)
        # gerekirse bir liste yarat
        if foto_id not in mapping:
            mapping[foto_id] = list()
        # açıklamaları kaydet
        mapping[foto_id].append(foto_aciklama)
    return mapping

# açıklamaları ayrıştır
aciklamalar = aciklamalari_yukle(doc)


# In[70]:


#şimdi, açıklama metnini temizlememiz gerekiyor. 
#Açıklamalar zaten belirtilmiş ve üzerinde çalışılması kolay.
#Çalışmamız gereken kelimelerin boyutunu azaltmak için metni aşağıdaki şekillerde temizleyeceğiz:
#tüm kelimeleri küçük harfe dönüştür
#noktalama işaretlerini kaldır
#bir karakter olan kelimeleri kaldır.
#içinde sayı olan tüm kelimeleri kaldır.



import string

def aciklama_temiz(aciklamalar):
    # noktalama işaretlerini kaldırmak için tablo hazırla
    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in aciklamalar.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            # tokenize
            desc = desc.split()
            # küçük harfe döndür
            desc = [word.lower() for word in desc]
            # noktalama işareti kaldır
            desc = [w.translate(table) for w in desc]
            desc = [word for word in desc if len(word)>1]
            # sayılı kelimeleri kaldır
            desc = [word for word in desc if word.isalpha()]
            # string olarak sakla
            desc_list[i] =  ' '.join(desc)

# temiz açıklamalar
aciklama_temiz(aciklamalar)


# In[71]:


#Temizlendikten sonra, kelime dağarcığının boyutunu özetleyebiliriz.

#İdeal olarak, hem anlamlı hem de mümkün olduğunca küçük bir kelime dağarcığı istiyoruz. 
#Daha küçük bir kelime dağarcığı, daha hızlı çalışacak daha küçük bir modelle sonuçlanacaktır.



# yüklenen açıklamaları bir kelime dağarcığına dönüştür
def kelime_dagarcigi(aciklamalar):
    # tüm açıklama dizelerinin bir listesini oluştur
    tum_aciklama = set()
    for key in aciklamalar.keys():
        [tum_aciklama.update(d.split()) for d in aciklamalar[key]]
    return tum_aciklama


# In[72]:


#her satırda bir resim tanımlayıcı ve açıklama ile descriptions.txt adlı yeni bir dosyaya kaydedelim.
def aciklamalari_kaydet(aciklamalar, dosyaismi):
    lines = list()
    for key, desc_list in aciklamalar.items():
        for desc in desc_list:
            lines.append(key + ' ' + desc)
    data = '\n'.join(lines)
    dosya = open(dosyaismi, 'w')
    dosya.write(data)
    dosya.close()


# In[73]:


dosyaismi = 'sss-2.txt'
# yükle
doc = dosya_yukle(dosyaismi)
# ayrıştır
aciklamalar = aciklamalari_yukle(doc)
# temizle
aciklama_temiz(aciklamalar)
# kelime haznesi
vocabulary = kelime_dagarcigi(aciklamalar)
print('Kelime Sayisi: %d' % len(vocabulary))
# kaydet
aciklamalari_kaydet(aciklamalar, 'descriptions.txt')

#açıklamaların temiz haline descriptions.txt dosyasına kaydettik.


# In[74]:


### DERİN ÖĞRENME MODELİ


# In[75]:


#hazırlanan fotoğraf ve metin verilerini modele ekle


from pickle import load

# hafızaya yükle
def dosya_yukle(dosyaismi):
    dosya = open(dosyaismi, 'r')
    text = dosya.read()
    dosya.close()
    return text

# önceden tanımlanmış fotoğraf tanımlayıcı listesini yükle
def set_yukle(dosyaismi):
    doc = dosya_yukle(dosyaismi)
    veriseti = list()
    # satır satır işle
    for line in doc.split('\n'):
        # boş satırları geç
        if len(line) < 1:
            continue
        # foto tanımlayıcısı
        identifier = line.split('.')[0]
        veriseti.append(identifier)
    return set(veriseti)

# temiz açıklamaları yükle
def temiz_aciklama_yukle(dosyaismi, veriseti):
    doc = dosya_yukle(dosyaismi)
    aciklamalar = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        foto_id, foto_aciklama = tokens[0], tokens[1:]
        # olmayan resimleri temizle
        if foto_id in veriseti:
            if foto_id not in aciklamalar:
                aciklamalar[foto_id] = list()
            desc = 'basla ' + ' '.join(foto_aciklama) + ' bitir'
            aciklamalar[foto_id].append(desc)
    return aciklamalar

# foto özelliklerini yükle
def foto_ozellik_yukle(dosyaismi, veriseti):
    tum_ozellikler = load(open(dosyaismi, 'rb'))
    # filtrele
    ozellikler = {k: tum_ozellikler[k] for k in veriseti}
    return ozellikler

# eğitim seti (6K)
dosyaismi = 'Flickr8k_text/Flickr_8k.trainImages.txt'
train = set_yukle(dosyaismi)
print('Veriseti: %d' % len(train))
# açıklamalar
egitim_aciklamalari = temiz_aciklama_yukle('descriptions.txt', train)
# foto özellikleri
egitim_ozellik = foto_ozellik_yukle('features.pkl', train)
print('Fotoğraflar: train=%d' % len(egitim_ozellik))


# In[76]:


def satirlara(aciklamalar):
    tum_aciklama = list()
    for key in aciklamalar.keys():
        [tum_aciklama.append(d) for d in aciklamalar[key]]
    return tum_aciklama

def belirtec_yarat(aciklamalar):
    lines = satirlara(aciklamalar)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def max_uzunluk(aciklamalar):
    lines = satirlara(aciklamalar)
    return max(len(d.split()) for d in lines)


# In[77]:


### MODELİ TANIMLA


# In[78]:


# altyazılama modelini tanımlayalım
def model_tanimla(vocab_size, max_uzunluk):
    # özellik çıkarma modeli
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    # sekans modeli
    inputs2 = Input(shape=(max_uzunluk,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    # decoder model
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # özet
    print(model.summary())
    plot_model(model, to_file='model.png', show_shapes=True)
    return model


# In[66]:


# bir görüntü için görüntü dizileri, giriş dizileri ve çıktı sözcükleri oluştur
def create_sequences(tokenizer, max_uzunluk, desc_list, photo, vocab_size):
    X1, X2, y = list(), list(), list()
    # resim için her açıklamayı gözden geçir
    for desc in desc_list:
        # encode
        seq = tokenizer.texts_to_sequences([desc])[0]
        # bir diziyi birden çok X,y çiftine böl
        for i in range(1, len(seq)):
            # giriş ve çıkış çiftine bölün
            in_seq, out_seq = seq[:i], seq[i]
            # pad giriş sekansı
            in_seq = pad_sequences([in_seq], maxlen=max_uzunluk)[0]
            # encode çıktı
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            # kaydet
            X1.append(photo)
            X2.append(in_seq)
            y.append(out_seq)
    return array(X1), array(X2), array(y)




# eğitim seti
 
# eğitim seti
dosyaismi = 'Flickr8k_text/Flickr_8k.trainImages.txt'
train = set_yukle(dosyaismi)
print('Veriseti: %d' % len(train))

egitim_aciklamalari = temiz_aciklama_yukle('descriptions.txt', train)


egitim_ozellik = foto_ozellik_yukle('features.pkl', train)
print('Fotoğraflar: train=%d' % len(egitim_ozellik))
tokenizer = belirtec_yarat(egitim_aciklamalari)
vocab_size = len(tokenizer.word_index) + 1
print('Kelime Sayısı: %d' % vocab_size)
# maksimum dizi uzunluğunu belirle
max_uzunluk = max_uzunluk(egitim_aciklamalari)
print('Aciklama Uzunluğu: %d' % max_uzunluk)
# dizileri hazırlamak
X1train, X2train, ytrain = create_sequences(tokenizer, max_uzunluk, egitim_aciklamalari, egitim_ozellik, vocab_size)
 
# dev dataset
 
# test seti yükle
dosyaismi = 'Flickr8k_text/Flickr_8k.devImages.txt'
test = set_yukle(dosyaismi)
print('Veriseti: %d' % len(test))
test_descriptions = temiz_aciklama_yukle('descriptions.txt', test)
# fotoğraf özellikleri
test_features = foto_ozellik_yukle('features.pkl', test)
print('Fotoğraflar: test=%d' % len(test_features))
# dizileri hazırlamak
X1test, X2test, ytest = create_sequences(tokenizer, max_uzunluk, test_descriptions, test_features, vocab_size)
 

model = model_tanimla(vocab_size, max_uzunluk)
# kontrol noktası geri aramasını tanımla

#Kodu yazarken eğitim veri setinde en iyi beceriye sahip kayıtlı modeli kullanmak için, 
#holdout geliştirme veri seti üzerinde eğitilen modelin becerisini izleyerek, 
#Epoch’un sonunda modelin geliştirme veri kümesindeki becerisi geliştiğinde, 
#tüm modeli bir dosyaya kaydettik.
#belli bir süre sonra modelin gelişimi durduğu için süreci durdurduk, ve en iyi modeli kaydederek modelimizi test aşamasına geçtik.
filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# fit model
model.fit([X1train, X2train], ytrain, epochs=20, verbose=2, callbacks=[checkpoint], validation_data=([X1test, X2test], ytest))


# In[64]:


#model 3. epoch'un sonunda eğitim verisinde 3.945 loss 
#geliştirme verisinde 4.243 loss değerine ulaştı ve kayıt edildi.

### MODELİ DEĞERLENDİR


# In[15]:


from numpy import argmax
from pickle import load
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu


def dosya_yukle(dosyaismi):
    dosya = open(dosyaismi, 'r')
    text = dosya.read()
    dosya.close()
    return text


def set_yukle(dosyaismi):
    doc = dosya_yukle(dosyaismi)
    veriseti = list()
    for line in doc.split('\n'):
        if len(line) < 1:
            continue
        identifier = line.split('.')[0]
        veriseti.append(identifier)
    return set(veriseti)


def temiz_aciklama_yukle(dosyaismi, veriseti):
    doc = dosya_yukle(dosyaismi)
    aciklamalar = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        foto_id, foto_aciklama = tokens[0], tokens[1:]
        if foto_id in veriseti:
            if foto_id not in aciklamalar:
                aciklamalar[foto_id] = list()
            desc = 'basla ' + ' '.join(foto_aciklama) + ' bitir'
            aciklamalar[foto_id].append(desc)
    return aciklamalar


def foto_ozellik_yukle(dosyaismi, veriseti):
    tum_ozellikler = load(open(dosyaismi, 'rb'))
    ozellikler = {k: tum_ozellikler[k] for k in veriseti}
    return ozellikler


def satirlara(aciklamalar):
    tum_aciklama = list()
    for key in aciklamalar.keys():
        [tum_aciklama.append(d) for d in aciklamalar[key]]
    return tum_aciklama


def belirtec_yarat(aciklamalar):
    lines = satirlara(aciklamalar)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


def max_uzunluk(aciklamalar):
    lines = satirlara(aciklamalar)
    return max(len(d.split()) for d in lines)


def kelime_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def generate_desc(model, tokenizer, photo, max_uzunluk):
    in_text = 'basla'
    for i in range(max_uzunluk):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_uzunluk)
        yhat = model.predict([photo,sequence], verbose=0)
        # olasılığı tam sayıya çevir
        yhat = argmax(yhat)
        # tamsayıyı kelimeye eşle
        word = kelime_id(yhat, tokenizer)
        # kelimeyi eşleyemezsek dur
        if word is None:
            break
        # sonraki kelimeyi oluşturmak için girdi olarak ekle
        in_text += ' ' + word
        # dizinin sonunu tahmin edersek dur
        if word == 'bitir':
            break
    return in_text

# modelin becerisini değerlendir
def modeli_degerlendir(model, aciklamalar, photos, tokenizer, max_uzunluk):
    gercek, predicted = list(), list()
    # tüm setin üzerinden geç
    for key, desc_list in aciklamalar.items():
        # açıklama oluştur
        yhat = generate_desc(model, tokenizer, photos[key], max_uzunluk)
        # gerçek ve tahmin edileni kaydet
        references = [d.split() for d in desc_list]
        gercek.append(references)
        predicted.append(yhat.split())
    print('BLEU-1: %f' % corpus_bleu(gercek, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(gercek, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(gercek, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(gercek, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

# eğitim setinde belirteç hazırla

# eğitim seti yükle
dosyaismi = 'Flickr8k_text/Flickr_8k.trainImages.txt'
train = set_yukle(dosyaismi)
print('Veriseti: %d' % len(train))
# açıklamalar
egitim_aciklamalari = temiz_aciklama_yukle('descriptions.txt', train)
# belirteç hazırla
tokenizer = belirtec_yarat(egitim_aciklamalari)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# maksimum dizi uzunluğunu belirle
max_uzunluk = max_uzunluk(egitim_aciklamalari)
print('Description Length: %d' % max_uzunluk)

# eğitim setini hazırla

# eğitim setini yükle
dosyaismi = 'Flickr8k_text/Flickr_8k.testImages.txt'
test = set_yukle(dosyaismi)
print('Veriseti: %d' % len(test))
# açıklamalar
test_descriptions = temiz_aciklama_yukle('descriptions.txt', test)
# foto özellikler
test_features = foto_ozellik_yukle('features.pkl', test)
print('Photos: test=%d' % len(test_features))

# modeli yükle
dosyaismi = 'model-ep003-loss3.945-val_loss4.243.h5'
model = load_model(dosyaismi)
# modeli değerlendir
modeli_degerlendir(model, test_descriptions, test_features, tokenizer, max_uzunluk)


# In[ ]:


### YENİ ALTYAZILAMALAR ÜRETELİM


# In[5]:


#yeni fotoğraflara altyazılama oluşturmak için ihtiyacımız olan hemen her şey model dosyasında.
#şimdi tokenizer'a ve modeli tanımlarken kullanılan maksimum dizi uzunluğuna ihtiyacımız var. bu bizim için 31.

from keras.preprocessing.text import Tokenizer
from pickle import dump


def dosya_yukle(dosyaismi):
    dosya = open(dosyaismi, 'r')
    text = dosya.read()
    dosya.close()
    return text

# önceden tanımlanmış  fotoğraf tanımlayıcı listesini yükle
def set_yukle(dosyaismi):
    doc = dosya_yukle(dosyaismi)
    veriseti = list()
    # satır satır işle
    for line in doc.split('\n'):
        # boş satırları geç
        if len(line) < 1:
            continue
        # resim tanımlayıcısını al
        identifier = line.split('.')[0]
        veriseti.append(identifier)
    return set(veriseti)

# temiz açıklamaları belleğe yükle
def temiz_aciklama_yukle(dosyaismi, veriseti):
    doc = dosya_yukle(dosyaismi)
    aciklamalar = dict()
    for line in doc.split('\n'):
        # satırı boşlukla bölme
        tokens = line.split()
        # kimliği açıklamadan ayır
        foto_id, foto_aciklama = tokens[0], tokens[1:]
        # sette olmayan resimleri atla
        if foto_id in veriseti:
            # liste oluştur
            if foto_id not in aciklamalar:
                aciklamalar[foto_id] = list()
            # açıklamayı belirteçlere kaydır
            desc = 'basla ' + ' '.join(foto_aciklama) + ' bitir'
            # kaydet
            aciklamalar[foto_id].append(desc)
    return aciklamalar

# temiz açıklamalar
def satirlara(aciklamalar):
    tum_aciklama = list()
    for key in aciklamalar.keys():
        [tum_aciklama.append(d) for d in aciklamalar[key]]
    return tum_aciklama

# verilen açıklamalarına bir tokenizer sığdır
def belirtec_yarat(aciklamalar):
    lines = satirlara(aciklamalar)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

# eğitim verisi yükle
dosyaismi = 'Flickr8k_text/Flickr_8k.trainImages.txt'
train = set_yukle(dosyaismi)
print('Veriseti: %d' % len(train))
egitim_aciklamalari = temiz_aciklama_yukle('descriptions.txt', train)
tokenizer = belirtec_yarat(egitim_aciklamalari)
dump(tokenizer, open('tokenizer.pkl', 'wb'))


# In[ ]:


#Artık, ek açıklamaların tüm eğitim veri kümesini yüklemek zorunda kalmadan, 
#tokenizer'ı ihtiyacımız olduğunda yükleyebiliriz.
#Şimdi yeni bir fotoğraf için bir açıklama oluşturalım.
#veri hazırlığında kullandığımız fonksiyonu burada tekrar kullanalım
#ancak bu sefer tek bir fotoğraf üzerinde çalışacak.


# In[98]:


from pickle import load
from numpy import argmax
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import load_model


def ozellik_ayikla(dosyaismi):
    
    model = VGG16()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    image = load_img(dosyaismi, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    return feature

# bir tamsayıyı bir kelimeye eşle
def kelime_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# açıklama üret
def generate_desc(model, tokenizer, photo, max_uzunluk):
    # üretme süreci baslatma
    in_text = 'basla'
    # dizinin tüm uzunluğu boyunca yinele
    for i in range(max_uzunluk):
        # tamsayı kodlama giriş sırası
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_uzunluk)
        # sonraki kelimeyi tahmin et
        yhat = model.predict([photo,sequence], verbose=0)
        # olasılığı tam sayıya çevir
        yhat = argmax(yhat)
        # tamsayıyı kelimeye eşle
        word = kelime_id(yhat, tokenizer)
        # kelimeyi eşleyemezsek dur
        if word is None:
            break
        # sonraki kelimeyi oluşturmak için girdi olarak ekle
        in_text += ' ' + word
        # dizinin sonunu tahmin edersek dur
        if word == 'bitir':
            break
    return in_text

tokenizer = load(open('tokenizer.pkl', 'rb'))
# eğitim verisinde tanımlanmış max uzunluk
max_uzunluk = 31
# modeli yükle
model = load_model('model-ep003-loss3.945-val_loss4.243.h5')
photo = ozellik_ayikla('bird.jpg')
# acıklama üret
aciklama = generate_desc(model, tokenizer, photo, max_uzunluk)
print(aciklama)


# In[ ]:




