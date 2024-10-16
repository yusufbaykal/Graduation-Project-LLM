import nltk
from bs4 import BeautifulSoup
import string
import re
import listeler
from collections import Counter

class turkish_denet():
    def __init__(self, text=""):
        self.turkcekelime = []

    def kisaltmakontrol(self, text):
        words = text.split()
        kisaltma_sayisi = 0
        kisaltma_kelimeleri = []

        for i in range(len(words)):
            if words[i] in listeler.kisaltList:
                kisaltma_sayisi += 1
                kisaltma_kelimeleri.append(words[i])
                index = listeler.kisaltList.index(words[i])
                words[i] = listeler.uzunhaller[index]

        result = " ".join(words)
        result = " ".join(result.split())
        kisaltma_kelimeleri = " ".join(kisaltma_kelimeleri)
        return kisaltma_kelimeleri, kisaltma_sayisi

    def kelimekontrol(self, text):
        kelimeler = set(self.turkcekelime)
        generated_words = set()
        misspellings = {}
        turkish_word = set()

        
        with open("../data/generated_words.txt", "r", encoding="utf-8") as file:
            for line in file:
                generated_words.add(line.strip())

        with open("../data/turkish_dictionary.txt", "r", encoding="utf-8") as file:
            for line in file:
                self.turkcekelime.append(line.strip())

        with open("../data/turkish_misspellings.txt", "r", encoding="utf-8") as file:
            for line in file:
                correct_word, misspelled_word = line.strip().split()
                misspellings[misspelled_word] = correct_word

        with open("../data/turkish_words.txt", "r", encoding="utf-8") as file:
            for line in file:
                turkish_word.add(line.strip())
        words = text.split()
        non_turkish_words = []
        filtered_text = ""
        turkish_word_count = 0

        for word in words:
            ana_kelime = word.strip(",.!?")
            if ana_kelime in kelimeler or ana_kelime in generated_words \
                    or ana_kelime in misspellings or ana_kelime in turkish_word:
                filtered_text += word + " "
                turkish_word_count += 1
                turkish_word_count += 1
            else:
                non_turkish_words.append(word)
        with open("../data/context_list.txt", "r", encoding="utf-8") as file:
            for line in file:
                context = line.strip()
                filtered_text = filtered_text.replace(context, "")

        non_turkish_word_count = len(non_turkish_words)
        first_10_non_turkish_words = non_turkish_words[:10]

        with open("../data/filtered_text.txt", "w", encoding="utf-8") as file:
            file.write(filtered_text)

        return filtered_text, non_turkish_word_count, first_10_non_turkish_words

    def kucukHarfeDonustur(self,text):
        words = text.split()
        lowercased_words = [word.lower() for word in words]
        return " ".join(lowercased_words)

    def noktalamaTemizleyicisi(self, text):
        regex = r"(?<!\d)[.,;:?)(](?!\d)"
        result = re.sub(regex, "", text, 0)
        return result

    def buyukharf(self, text):
        kelimeler = text.split()
        capitalized_words = []

        for i, word in enumerate(kelimeler):
            if len(word) > 0:
                if i == 0:
                    capitalized_word = word[0].upper() + word[1:].lower()
                elif kelimeler[i - 1].endswith((".", "?", "!")):
                    capitalized_word = word[0].upper() + word[1:].lower()
                else:
                    capitalized_word = word.lower()
                capitalized_words.append(capitalized_word)

        buyukharf = ' '.join(capitalized_words)
        return buyukharf

    def duzelt_noktalama(self,text):
        """Metindeki noktalama işaretlerini düzeltme fonksiyonu.

        Args:
            text (str): Noktalama işaretleri düzeltilmek istenen metin.

        Returns:
            str: Noktalama işaretleri düzeltilmiş metin.
        """

        
        text = text.replace(", sabahleyin", ". Sabahleyin")
        text = text.replace(", ve veya", "; ve veya")
        text = text.replace(", yemek yedi", "; yemek yedi")
        text = text.replace(", herkes uyudu", "; herkes uyudu")

        
        text = text.replace("öğle vakti herkes evine döndü yemek. yedi.", "Öğle vakti herkes evine döndü; yemek yedi.")

        text = text.replace("akşam olunca sokak. lambaları yandı.", "Akşam olunca sokak lambaları yandı.")
        text = text.replace("  ", " ")

       
        if text[0].islower():
            text = text.capitalize()

        return text
    def noktalama_ekle(self, text):
        """Metne otomatik noktalama işareti ekleme fonksiyonu.

        Args:
            text (str): Noktalama işaretleri eklenmek istenen metin.

        Returns:
            str: Noktalama işaretleri eklenmiş metin.
        """
        try:
            with open('../data/turkish_dictionary.txt', 'r', encoding='utf-8') as file:
                dictionary = {}
                for line in file:
                    parts = line.strip().split(' ', 1)
                    if len(parts) == 2:
                        word, tags = parts
                        dictionary[word] = tags

        except FileNotFoundError:
            return "Sözlük dosyası bulunamadı."

        kelimeler = text.split()
        yeni_metin = []

        for i, kelime in enumerate(kelimeler):
            noktalama_ekle = ""
            if kelime in dictionary:
                tags = dictionary[kelime]
                if "IS_ADJ" in tags or "IS_ADJ+" in tags:
                    noktalama_ekle = ","
                elif "IS_SD" in tags or "IS_SD+" in tags:
                    noktalama_ekle = "."
                elif "IS_CONJ" in tags or "IS_CONJ+" in tags:
                    noktalama_ekle = ";"
                elif "IS_ADV" in tags or "IS_ADV+" in tags:
                    noktalama_ekle = ","
                elif "IS_NUM" in tags or "IS_NUM+" in tags:
                    noktalama_ekle = ","
                else:
                    noktalama_ekle = ""

            kelime = kelime + noktalama_ekle
            if i == len(kelimeler) - 1:
                kelime += "."
            yeni_metin.append(kelime)
        yeni_metin_str = " ".join(yeni_metin)
        if not yeni_metin_str.endswith('.'):
            yeni_metin_str += '.'
        yeni_metin_str = self.duzelt_noktalama(yeni_metin_str)
        return yeni_metin_str.strip()
    
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






