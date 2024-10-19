import nltk
from bs4 import BeautifulSoup
import string
import re
import listeler
from collections import Counter


class TurkishNLP():
    def enCokKelime(self, text):
        kelimeAyir = nltk.FreqDist(text.lower().split())
        en_cok_kelimeler = kelimeAyir.most_common(20)
        kelime_listesi = []
        for kelime, sayi in en_cok_kelimeler:
            kelime_listesi.append(f"{kelime}: {sayi}")
        return kelime_listesi

    def alfaNumerik(self, text):
        numerikKarakter = sum(len(w) for w in text.split() if not w.isalnum())
        numerikKarakterYuzdesi = numerikKarakter / len(text.replace(" ", ""))
        return numerikKarakter, numerikKarakterYuzdesi

    def harfDonusum(self,text):
        text = re.sub(r"Â", "A", text)
        text = re.sub(r"â", "a", text)
        text = re.sub(r"Î", "I", text)
        text = re.sub(r"î", "ı", text)
        text = re.sub(r"Û", "U", text)
        text = re.sub(r"û", "u", text)

        return text

    def metin_istatistik(self, text):
        kelimeler = text.split()
        kelime_sayisi = len(kelimeler)
        karakter_sayisi = len(text)
        benzersiz_kelime_sayisi = len(set(kelimeler))
        kelime_frekanslari = Counter(kelimeler)
        en_cok_kelimeler_list = kelime_frekanslari.most_common(5)
        en_cok_kelimeler = [kelime[0] for kelime in en_cok_kelimeler_list]
        en_az_kelimeler_list = kelime_frekanslari.most_common()[:-6:-1]
        en_az_kelimeler = [kelime[0] for kelime in en_az_kelimeler_list if kelime[0] not in string.punctuation]
        ortalama_kelime_uzunlugu = sum(len(kelime) for kelime in kelimeler) / kelime_sayisi
        en_uzun_kelime = max(kelimeler, key=len)
        en_kisa_kelime = min((kelime for kelime in kelimeler if kelime not in string.punctuation), key=len, default="")
        kelime_yogunlugu = kelime_sayisi / karakter_sayisi

        return {
            'Kelime Sayısı': kelime_sayisi,
            'Karakter Sayısı': karakter_sayisi,
            'Benzersiz Kelime Sayısı': benzersiz_kelime_sayisi,
            'En Çok Kullanılan Kelimeler': en_cok_kelimeler,
            'En Az Kullanılan Kelimeler': en_az_kelimeler,
            'Ortalama Kelime Uzunluğu': ortalama_kelime_uzunlugu,
            'En Uzun Kelime': en_uzun_kelime,
            'En Kısa Kelime': en_kisa_kelime,
            'Kelime Yoğunluğu': kelime_yogunlugu
        }


    def kucukHarfeDonustur(text):
        return text.lower()

    def noktalamaIsaretleriniKaldir(self, text):

        text = re.sub(r"\n", " ", text)
        text = re.sub(r"  ", " ", text)

        text = re.sub(r"I", "ı", text)
        text = kucukHarfeDonustur(text)
        text = noktalamaIsaretleriniKaldir(text)
        return re.sub(r'[^\w\s]', '', text)

    def noktalamaTemizleyicisi(self,text):
        regex = r"(?<!\d)[.,;:?)(](?!\d)"
        result = re.sub(regex, "", text, 0)
        return result

    def stopKelimeleriKaldir(self, text):
        with open('../data/stopwords.txt', 'r', encoding='utf-8') as f:
            stopKelimeler = set(f.read().splitlines())

        kaldırılan_kelimeler = []
        temizlenmis_metin = []

        for kelime in text.split():
            kelime_lower = kelime.lower()
            if kelime_lower not in stopKelimeler:
                temizlenmis_metin.append(kelime)
            else:
                kaldırılan_kelimeler.append(kelime)

        temizlenmis_metin = ' '.join(temizlenmis_metin)
        kaldırılan_kelimeler_str = ', '.join(kaldırılan_kelimeler)

        print("Kaldırılan Stop Word Kelimeleri:", kaldırılan_kelimeler_str)
        return temizlenmis_metin, kaldırılan_kelimeler

    def htmlEtiketleriniKaldir(self,html):
        soup = BeautifulSoup(html, 'html.parser')
        text = soup.get_text(" ")
        return text

    def kisaltmakontrol(self, text):
        kisaltList = listeler.kisaltList
        uzunhaller = listeler.uzunhaller
        words = text.split()
        for i in range(len(words)):
            if words[i] in kisaltList:
                index = kisaltList.index(words[i])
                words[i] = uzunhaller[index]
        result = " ".join(words)
        result = " ".join(result.split())
        return result


    def clean_text(self, text):
        punctuation_remove = [r"\.", r"'", r"!", r",", r"\?"]
        for punctuation in punctuation_remove:
            text = re.sub(punctuation, "", text)

        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'\n', ' ', text)
        text = re.sub(r'\t', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r"#(\w+)", '', text)
        text = re.sub(r"@(\w+)", '', text)
        text = re.sub(r'\d', '', text)
        text = text.strip().lower()
        
        text = text.replace('&nbsp;', ' ').replace('“', ' ').replace('·', ' ').replace('i̇', 'i')
        text = text.replace('•', ' ').replace('\xa0', ' ')
        text = text.replace('”', ' ').replace('nan', '').replace('\r', '').replace('’', ' ')

        with open('../data/stopwords.txt', 'r', encoding='utf-8') as f:
            stopwordsList = set(f.read().splitlines())
        text = " ".join([word for word in text.split() if word not in stopwordsList])

        return text


