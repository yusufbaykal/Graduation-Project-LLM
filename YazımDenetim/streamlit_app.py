import os
import pandas as pd
import streamlit as st
from turkish_yaz import turkish_denet
from turkish_yaz import TurkishNLP
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.callbacks.manager import get_openai_callback
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import WebBaseLoader
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)
from langchain_community.vectorstores import Chroma
import json
from streamlit_lottie import st_lottie
from PIL import Image
from dotenv import load_dotenv
denetci = turkish_denet()
turknlp = TurkishNLP()

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def main():
    st.set_page_config(
        layout="wide",
        page_title="Pi. Lab Teknoloji",
        page_icon="❄️",
    )

    page_map = {
        "Ana Sayfa": AnaSayfa,
        "Metin İstatistik": Metin_İstatistik,
        "Model Hazırlık": ModelHazırlık,
        "PDF Chat-Bot": chat_pdf,
        "Web Chat-Bot": chat_web,
        "Analizler & Veri Ön İşleme": turkish_data_preprocessing,
    }
    page = st.sidebar.selectbox("#### Sayfa Seçiniz", list(page_map.keys()))

    if page:
        page_map[page]()

    st.sidebar.write(
        """
        #### Contact:
        - [LinkedIn](https://www.linkedin.com/in/yusufbaykaloglu/)
        - [Github](https://github.com/yusufbaykal)
        - [Medium](https://medium.com/@yusufbaykaloglu)
        """
    )

def AnaSayfa():
    col1, col2 = st.columns([0.25, 0.75])
    with col1:
        st.markdown(
            """
            <h1 style="font-size: 48px; font-weight: bold; color: #000000; margin-top: 0px; margin-bottom: 10px;">Pi. Lab Teknoloji</h1>
            <p style="font-size: 18px; font-weight: normal; color: #666666;">Yapay zeka ve veri bilimi ile işinizi büyütün.</p>
            """,
            unsafe_allow_html=True,
        )
        logo = Image.open(r"../Images/logo.png")
        st.image(logo, width=150)

        st.markdown(
            """
            <p style="font-size: 16px; font-weight: normal; color: #333333;">
            Pi. Lab Teknoloji, yapay zeka ve veri bilimi alanlarında uzmanlaşmış bir şirkettir. İşletmenizin ihtiyaçlarını karşılamak için özel çözümler sunuyoruz.
            </p>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown("""
        <div style="display: flex; justify-content: space-evenly; margin-top: 50px;">
            <a href="#Metin_İstatistik" target="_self">
                <button style="background-color: #f7f7f7; border-radius: 10px; padding: 20px; font-size: 18px; font-weight: normal; color: #000000; width: 200px; text-align: center;">Metin İstatistik</button>
            </a>
            <a href="#ModelHazırlık" target="_self">
                <button style="background-color: #f7f7f7; border-radius: 10px; padding: 20px; font-size: 18px; font-weight: normal; color: #000000; width: 200px; text-align: center;">Model Hazırlık</button>
            </a>
        </div>
        <div style="display: flex; justify-content: space-evenly; margin-top: 50px;">
            <a href="#chat_pdf" target="_self">
                <button style="background-color: #f7f7f7; border-radius: 10px; padding: 20px; font-size: 18px; font-weight: normal; color: #000000; width: 200px; text-align: center;">PDF Chatbot</button>
            </a>
            <a href="#chat_web" target="_self">
                <button style="background-color: #f7f7f7; border-radius: 10px; padding: 20px; font-size: 18px; font-weight: normal; color: #000000; width: 200px; text-align: center;">Web Chatbot</button>
            </a>
            <a href="#analizler_veri_on_isleme" target="_self">
                <button style="background-color: #f7f7f7; border-radius: 10px; padding: 20px; font-size: 18px; font-weight: normal; color: #000000; width: 200px; text-align: center;">Analizler & Veri Ön İşleme</button>
            </a>
        </div>
        """, unsafe_allow_html=True)

def chat_pdf():
    st.header("Dökümanlarınız İle Sohbet Edin!! 🤷‍♀️💬")

    pdf = st.sidebar.file_uploader("Döküman Yükle", type="pdf")
    text = ""
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    if text:
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        if chunks:
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            knowledge_base = FAISS.from_texts(chunks, embeddings)

            with st.form(key='my_form'):
                st.write(" Merhabalar, Sorularınızı Sormaya Devam Edin. 👋")
                user_question = st.text_input("Sorunuzu Giriniz:")
                if st.form_submit_button("Soru Gönder"):
                    if user_question:
                        docs = knowledge_base.similarity_search(user_question)
                        llm = ChatOpenAI(model_name="gpt-3.5-turbo")
                        chain = load_qa_chain(llm, chain_type="stuff")
                        with get_openai_callback():
                            response = chain.run(input_documents=docs, question=user_question)
                        st.write(response)
        else:
            st.error("Döküman metni boş. Lütfen geçerli bir PDF yükleyin.")

def chat_web():
    st.subheader('🦜🔗 Web Chatbot 🦜🔗')
    url = st.text_input("##### Web Sitesini Giriniz:")

    prompt = st.text_area("##### Sorularınızı Giriniz:")
    if st.button("Soruları Gönder", type="primary"):
        ABS_PATH = os.path.dirname(os.path.abspath(__file__))
        DB_DIR = os.path.join(ABS_PATH, "db")

        loader = WebBaseLoader(url)
        data = loader.load()

        text_splitter = CharacterTextSplitter(
            separator='\n',
            chunk_size=500,
            chunk_overlap=40
        )

        docs = text_splitter.split_documents(data)
        openai_embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=openai_embeddings,
            persist_directory=DB_DIR
        )

        vectordb.persist()
        retriever = vectordb.as_retriever()

        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model_name='gpt-3.5-turbo'),
            chain_type='stuff',
            retriever=retriever,
            return_source_documents=True
        )

        response = qa_chain(prompt)
        st.write(response)

def turkish_data_preprocessing():

    def import_json(path):
        with open(path, "r", encoding="utf8", errors="ignore") as file:
            data = json.load(file)
            return data

    robo_chat = import_json(r"../data/data.json")
    st_lottie(robo_chat, height=400, key="adv_chat")

    st.subheader("Data Preprocessing")
    st.markdown(
        """
        <style>
        .css-1l02zno {
            position: absolute;
            top: 0;
            right: 0;
        }
        </style>
        <a href="https://github.com/yusufbaykal/TurkceYazimDenetim" target="_blank" class="css-1l02zno">
            <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub Logo" width="30" height="30">
        </a>
        """,
        unsafe_allow_html=True,
    )

    menu = ["Kısaltma Kontrol", "Kelime Kontrol", "Noktalama İşareti Ekle", "HTML Etiketleri Temizleme",
            "En Çok Kullanılan Kelimeler", "Alfa-Numeric", "Harf Dönüşümü", "Türkçe Karakter Olmayan",
            "Noktalama İşareti Kaldır", "Stop-Words Kaldır", "Kelime İstatistikleri"]

    selected_menu = st.sidebar.selectbox("İşlem Seçiniz", menu)

    text = st.text_area("##### Metin Giriniz:")
    if st.button("## Denetle"):
        if text:
            if selected_menu == "Kısaltma Kontrol":
                text = denetci.kisaltmakontrol(text)
            elif selected_menu == "Kelime Kontrol":
                text,non_turkish_word_count,first_10_non_turkish_words = denetci.kelimekontrol(text)
                st.write("Türkçe Kelime Olmayan Sayısı:", non_turkish_word_count)
                st.write("10 Kelime:",first_10_non_turkish_words)
            elif selected_menu == "Noktalama İşareti Ekle":
                text = denetci.noktalama_ekle(text)
            elif selected_menu == "HTML Etiketleri Temizleme":
                text = denetci.htmlEtiketleriniKaldir(text)
            elif selected_menu == "En Çok Kullanılan Kelimeler":
                encokullanılankelimer = turknlp.enCokKelime(text)
                st.write("En Çok Kullanılan Kelimeler:")
                for i in encokullanılankelimer:
                 st.write("*",i)
            elif selected_menu == "Alfa-Numeric":
                numerikKarakter, numerikKarakterYuzdesi = turknlp.alfaNumerik(text)
                st.write("Toplamda Karakter Sayılmayan Değişken Sayısı:", numerikKarakter)
                st.write("Toplamda Karakter olmayanların Total Metin içerisindeki Yüzdesi:", numerikKarakterYuzdesi)
            elif selected_menu == "Harf Dönüşümü":
                text = turknlp.harfDonusum(text)
            elif selected_menu == "Noktalama İşareti Kaldır":
                text = turknlp.noktalamaTemizleyicisi(text)
            elif selected_menu == "Stop-Words Kaldır":
                temizlenmis_metin = turknlp.stopKelimeleriKaldir(text)
                kalan_kelime_sayisi = len(temizlenmis_metin.split())
                stop_kelime_sayisi = len(text.split()) - kalan_kelime_sayisi
                st.write("**Kaldırılan Stop Words Sayısı**:", stop_kelime_sayisi)
                st.write("**Kalan Kelime Sayısı**:", kalan_kelime_sayisi)
                st.write("**Kalan Metin**:", temizlenmis_metin)
                st.write("**Stop-Words Kaldırılmış Metin**:", temizlenmis_metin)
            else:
                st.warning("Geçerli bir işlem seçilmedi.")
                st.stop()
            #st.write("Denetleme Sonucu")
            #st.write(text)

        else:
            st.warning("Metin Girişi Gerçekleştirmediniz.")

    return

def Metin_İstatistik():
    """
    Genel Açıklama: Aşağıdaki kod bloğunda, kullanıcıdan alınan metin verisi üzerinde istatistik analizleri gerçekleştiriliyor.
    Kullanıcıdan alınan metin verisi üzerinde, metin istatistikleri hesaplanıyor. Hesaplanan metin istatistikleri tablo halinde
    kullanıcıya sunuluyor.
    """

    def import_json(path):
        with open(path, "r", encoding="utf8", errors="ignore") as file:
            data = json.load(file)
            return data

    robo_chat = import_json(r"../data/statistik.json")
    st_lottie(robo_chat, height=400, key="adv_chat")

    st.markdown("""
        #### Metin istatistikleri ile verilerinizin sesini dinleyin, gizemli anlamlarını açığa çıkarın ve bilinmeyenleri keşfedin!
        """)
    uploudData = st.file_uploader(
        "##### Txt Dosyasını Yükleyiniz.",
        type=["TXT"],
        help="##### Yüklenilen Belge Formata Uygun Değil!",
    )
    try:
        if uploudData is not None:
            text = uploudData.read().decode("utf-8")
            istatistik_text = turknlp.metin_istatistik(text)
            st.table([istatistik_text])
    except Exception as e:
        st.error(f"Hata: {e}")

def df_donustur(df):
    return df.to_csv().encode('utf-8')

def ModelHazırlık():

    def import_json(path):
        with open(path, "r", encoding="utf8", errors="ignore") as file:
            data = json.load(file)
            return data

    robo_chat = import_json(r"../data/data_isle.json")
    st_lottie(robo_chat, height=400, key="adv_chat")

    uploudData = st.file_uploader(
        "##### Veri Setinizi Yükleyiniz.",
        type=["CSV"],
        help="Yüklenilen Belge Formata Uygun Değil!",
    )
    st.markdown("""
    #### Veri Setinize Uygulanacak Fonksiyonlar
    
    - **Stop-Words Kaldırma**
    - **Noktalama İşaretleri Kaldırma**
    - **HTML Etiketleri Kaldırma**
    - **Alfa-Numeric İfadeleri Kaldırma**
    - **Türkçeye Uygun Harf Dönüşümleri**

    **Not**: Metin istatistiklerini Analiz ve Veri Ön İşleme Sayfasından alabilirsiniz.
    """)
    try:
        if uploudData is not None:
            df = pd.read_csv(uploudData)
            if "text" in df.columns:
                text = ' '.join(df["text"].head(5))
                st.write(text)
                st.markdown("Düzenlenmiş Metin")
                temiz = turknlp.clean_text(text)
                st.write(temiz)
                df_sonuc = pd.DataFrame({"Temizlenmiş Metin": [temiz]})
                csv = df_donustur(df_sonuc)
                st.session_state["temizlenmis_metin"] = temiz
                dosya_adi = st.text_input("Dosya Adı Giriniz:")
                if st.button("Dosyayı İndir"):
                    st.download_button(
                        label="Temizlenmiş Metni İndir",
                        data=csv,
                        file_name=dosya_adi + ".csv",
                        mime='text/csv',
                    )
            else:
                st.warning("Veri setinde 'text' sütunu bulunamadı.")
    except Exception as e:
        st.error(f"Hata: {e}")

if __name__ == '__main__':
    main()
