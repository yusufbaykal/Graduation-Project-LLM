listeler = ["IS_SD", "IS_KG", "IS_UD", "IS_UU", "IS_UUU",
            "IS_SU", "IS_ST", "F_SD", "F_GUD", "F_GUDO", "IS_SDD","CL_FIIL",
            "F1P1", "F2P1", "F2PL", "F2P1-NO-REF", "F3P1-NO-REF", "CL_ISIM",
            "F4P1-NO-REF", "F4PR-NO-REF", "F4PL-NO-REF", "F4PW-NO-REF", "F5PL-NO-REF",
            "F5PR-NO-REF", "F5PW-NO-REF", "F2P1", "F3P1", "F4P1","IS_BILEŞ",
            "F4PR", "F4PL", "F4PW", "F5P1", "F5PL","F_GUD",
            "F5PR", "F5PW", "F6P1", "IS_KU", "IS_BILEŞ",
            "IS_B_SD", "IS_KI", "IS_STT", "IS_UDD", "IS_CA", "IS_KIS",
            "IS_EX", "CL_NONE", "IS_B_SI", "IS_SAYI"]

with open('../data/turkish_dictionary.txt', 'r', encoding='utf-8') as file:
    veri = file.readlines()
    degerler = {deger: 0 for deger in listeler}

    for satir in veri:
        for deger in listeler:
            if deger in satir:
                degerler[deger] += 1

    for deger, adet in degerler.items():
        print(f"{deger}: {adet} Adet var")

print("IS_SD: İsim-fiil edatıdır.")
print("IS_KG: İsim-kök gelir edatıdır.")
print("IS_UD: İsim-ünlem dolayısıdır.")
print("IS_UU: İsim-ünlem ünlemidir.")
print("IS_UUU: İsim-ünlem-ünlem dolayısıdır.")
print("IS_SU: İsim-sıfat unsurudur.")
print("IS_ST: İsim-sıfat takısıdır.")
print("F_SD: Fiil-dolayısıdır.")
print("F_GUD: Fiil-gereksiz unsurdur.")
print("F_GUDO: Fiil-gereksiz unsur olumsuzdur.")
print("IS_SDD: İsim-dolayısıdır.")
print("CL_FIIL: Cümlede fiil olarak kullanılan kelimeyi temsil eder.")
print("F1P1: Birinci şahıs çoğul şimdiki zaman fiilidir.")
print("F2P1: İkinci şahıs çoğul şimdiki zaman fiilidir.")
print("F2PL: İkinci şahıs çoğul şimdiki zaman nesnesidir.")
print("F2P1-NO-REF: İkinci şahıs çoğul şimdiki zaman fiilidir, atıfta bulunmaz.")
print("F3P1-NO-REF: Üçüncü şahıs çoğul şimdiki zaman fiilidir, atıfta bulunmaz.")
print("CL_ISIM: Cümlede isim olarak kullanılan kelimeyi temsil eder.")
print("F4P1-NO-REF: Dördüncü şahıs çoğul şimdiki zaman fiilidir, atıfta bulunmaz.")
print("F4PR-NO-REF: Dördüncü şahıs çoğul şimdiki zaman dolayısıdır, atıfta bulunmaz.")
print("F4PL-NO-REF: Dördüncü şahıs çoğul şimdiki zaman nesnesidir, atıfta bulunmaz.")
print("F4PW-NO-REF: Dördüncü şahıs çoğul şimdiki zaman yalnızca olumsuzdur, atıfta bulunmaz.")
print("F5PL-NO-REF: Beşinci şahıs çoğul şimdiki zaman nesnesidir, atıfta bulunmaz.")
print("F5PR-NO-REF: Beşinci şahıs çoğul şimdiki zaman dolayısıdır, atıfta bulunmaz.")
print("F5PW-NO-REF: Beşinci şahıs çoğul şimdiki zaman yalnızca olumsuzdur, atıfta bulunmaz.")
print("F6P1: Altıncı şahıs çoğul şimdiki zaman fiilidir.")
print("IS_KU: İsim-kökün durumu belirtir.")
print("IS_BILEŞ: İsim-bileşik kelime unsurudur.")
print("IS_B_SD:İsim-bileşik dolayısıdır.")
print("IS_KI: İsim-kip edatıdır.")


