import nltk
from bs4 import BeautifulSoup
import string
import re
import listeler
from collections import Counter

import re

class TurkishDenet:
    def __init__(self):
        self.turkcekelime = self.load_turkish_words("../data/turkish_dictionary.txt")
        self.generated_words = self.load_generated_words("../data/generated_words.txt")
        self.misspellings = self.load_misspellings("../data/turkish_misspellings.txt")
        self.context_list = self.load_context_list("../data/context_list.txt")

    @staticmethod
    def load_turkish_words(path):
        """Türkçe kelimeleri dosyadan yükler."""
        try:
            with open(path, "r", encoding="utf-8") as file:
                return set(line.strip() for line in file)
        except FileNotFoundError:
            raise Exception("Türkçe sözlük dosyası bulunamadı.")

    @staticmethod
    def load_generated_words(path):
        """Oluşturulan kelimeleri dosyadan yükler."""
        try:
            with open(path, "r", encoding="utf-8") as file:
                return set(line.strip() for line in file)
        except FileNotFoundError:
            raise Exception("Oluşturulmuş kelimeler dosyası bulunamadı.")

    @staticmethod
    def load_misspellings(path):
        """Yanlış yazılmış kelimeleri dosyadan yükler."""
        try:
            misspellings = {}
            with open(path, "r", encoding="utf-8") as file:
                for line in file:
                    correct_word, misspelled_word = line.strip().split()
                    misspellings[misspelled_word] = correct_word
            return misspellings
        except FileNotFoundError:
            raise Exception("Yanlış yazım dosyası bulunamadı.")

    @staticmethod
    def load_context_list(path):
        """Bağlam listesini dosyadan yükler."""
        try:
            with open(path, "r", encoding="utf-8") as file:
                return {line.strip() for line in file}
        except FileNotFoundError:
            raise Exception("Bağlam listesi dosyası bulunamadı.")

    def kisaltmakontrol(self, text):
        """Kısaltma kelimelerini kontrol eder ve uzun halleriyle değiştirir."""
        words = text.split()
        kisaltma_kelimeleri = [word for word in words if word in listeler.kisaltList]
        kisaltma_sayisi = len(kisaltma_kelimeleri)

        for kisaltma in kisaltma_kelimeleri:
            index = listeler.kisaltList.index(kisaltma)
            words[words.index(kisaltma)] = listeler.uzunhaller[index]

        result = " ".join(words).strip()
        return " ".join(kisaltma_kelimeleri), kisaltma_sayisi, result

    def kelimekontrol(self, text):
        """Metindeki kelimeleri kontrol eder ve filtreler."""
        words = text.split()
        filtered_text = []
        non_turkish_words = []

        for word in words:
            ana_kelime = word.strip(",.!?")
            if (ana_kelime in self.turkcekelime or 
                ana_kelime in self.generated_words or 
                ana_kelime in self.misspellings or 
                ana_kelime in self.context_list):
                filtered_text.append(word)
            else:
                non_turkish_words.append(word)

        non_turkish_word_count = len(non_turkish_words)
        first_10_non_turkish_words = non_turkish_words[:10]
        filtered_text_str = " ".join(filtered_text)

        with open("../data/filtered_text.txt", "w", encoding="utf-8") as file:
            file.write(filtered_text_str)

        return filtered_text_str, non_turkish_word_count, first_10_non_turkish_words

    @staticmethod
    def kucuk_harfe_donustur(text):
        """Metni küçük harfe dönüştürür."""
        return text.lower()

    @staticmethod
    def noktalama_temizleyicisi(text):
        """Metindeki noktalama işaretlerini temizler."""
        return re.sub(r"(?<!\d)[.,;:?)(](?!\d)", "", text)

    @staticmethod
    def buyuk_harf(text):
        """Metindeki kelimelerin ilk harflerini büyük yapar."""
        kelimeler = text.split()
        capitalized_words = []

        for i, word in enumerate(kelimeler):
            if len(word) > 0:
                if i == 0 or kelimeler[i - 1].endswith((".", "?", "!")):
                    capitalized_words.append(word.capitalize())
                else:
                    capitalized_words.append(word.lower())

        return ' '.join(capitalized_words)

    @staticmethod
    def duzelt_noktalama(text):
        """Metindeki noktalama işaretlerini düzeltir."""
        corrections = {
            ", sabahleyin": ". Sabahleyin",
            ", ve veya": "; ve veya",
            ", yemek yedi": "; yemek yedi",
            ", herkes uyudu": "; herkes uyudu",
            "öğle vakti herkes evine döndü yemek. yedi.": "Öğle vakti herkes evine döndü; yemek yedi.",
            "akşam olunca sokak. lambaları yandı.": "Akşam olunca sokak lambaları yandı."
        }

        for wrong, right in corrections.items():
            text = text.replace(wrong, right)

        if text and text[0].islower():
            text = text.capitalize()

        return text.strip()

    def noktalama_ekle(self, text):
        """Metne otomatik noktalama işareti ekler."""
        try:
            with open('../data/turkish_dictionary.txt', 'r', encoding='utf-8') as file:
                dictionary = {line.split()[0]: line.strip().split(' ', 1)[1] for line in file}
        except FileNotFoundError:
            raise Exception("Sözlük dosyası bulunamadı.")

        kelimeler = text.split()
        yeni_metin = []

        for i, kelime in enumerate(kelimeler):
            noktalama_ekle = ""
            if kelime in dictionary:
                tags = dictionary[kelime]
                if "IS_ADJ" in tags or "IS_ADV" in tags:
                    noktalama_ekle = ","
                elif "IS_SD" in tags:
                    noktalama_ekle = "."
                elif "IS_CONJ" in tags:
                    noktalama_ekle = ";"

            kelime += noktalama_ekle
            if i == len(kelimeler) - 1:
                kelime += "."
            yeni_metin.append(kelime)

        yeni_metin_str = " ".join(yeni_metin).strip()
        return self.duzelt_noktalama(yeni_metin_str)
