import zipfile

zip_ref = zipfile.ZipFile('C:\\Users\\user\\PycharmProjects\\Deep_Learning\\Klasifikasi Berita Hoax Indonesia Word2Vec\\Indonesia word2vec embbeding.zip', 'r')
zip_ref.extractall('C:\\Users\\user\\PycharmProjects\\Deep_Learning\\Klasifikasi Berita Hoax Indonesia Word2Vec\\Indo Word2Vec')
zip_ref.close()