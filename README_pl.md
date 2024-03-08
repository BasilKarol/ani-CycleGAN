# <a name="x7610a18f0264f1cb403f046ccd5b781168b5c4d"></a>**AniCycleGAN - Projekt końcowy z kursu Deep Learning School prowadzonego przez Wasilija Korola**
### <a name="xf16f7aa3cc4d4f7ee61a8f5f5d2fa9bbadc173c"></a>**Projekt generowania i zmiany stylu obrazów z wykorzystaniem architektury CycleGAN**
## <a name="оглавление"></a>**Spis treści:**
- [Specyfikacja](#тех-инф)
- [Stosowanie](#запуск)
- [Efekty wykonanej pracy](#результаты)
- [Przegląd interfejsu graficznego](#результаты)

## <a name="техническая-информация"></a>**Specyfikacja**
Do stworzenia projektu użyliśmy: \* python 3.10 \* notatnik jupyter (kaggle)

Wszystkie biblioteki opisane są w pliku **WYMAGANIA.txt**

## <a name="использование"></a>**Stosowanie**
Aby uruchomić program GUI, wystarczy uruchomić główny plik w katalogu projektu za pomocą Pythona (Windows 10):

twój\folder\z\głównym\plikiem> python main.py

## <a name="результаты-проделанной-работы"></a>**Efekty wykonanej pracy**
Podstawą jest oficjalny artykuł „ ***Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Network 2017*** ” (https://arxiv.org/pdf/1703.10593.pdf) Wdrożenie sieci rozpoczęło się od zadania Monet2Photo – naucz sieć przenoszenia zdjęć na styl obrazów Moneta / nadaj obrazom Moneta fotorealizmu. Praca ta jest opisana w notatniku **cyklgan\_demo.ipynb**

*Wyniki cyklgan\_demo Monet na zdjęciach* : Wyniki cyklgan\_demo1

Wyniki cyklgan\_demo2

*I odwrotnie* : Wyniki cyklgan\_demo3

wyniki cyklgan\_demo4

**Głównym zadaniem** jest przekształcenie prawdziwej twarzy w odpowiadającą jej twarz w stylu japońskiej animacji z zachowaniem oryginalnych cech. Podstawowe szkolenie i architektura sieci są opisane w pliku **AniCycleGAN.ipynb** . Początkowo sieć była szkolona na zestawie danych Flickr Faces (https://www.kaggle.com/xhlulu/flickrfaceshq-dataset-nvidia-resized-256px). Jednak ze względu na różnorodność płci/wieku osób, a także różnicę w ich skali, sieć NIE była w stanie poprawnie przekazać stylu. Po poszukiwaniach znaleziono zbiór danych selfie2anime, który idealnie nadawał się do tego zadania (https://www.kaggle.com/arnaud58/selfie2anime).Szkolenie przeprowadzono przy użyciu karty graficznej Nvidia Tesla K80 dedykowanej dla platformy kaggle. Całe szkolenie trwało około 30 godzin. W finalnej wersji w 80% przypadków sieć skutecznie przenosi styl, a także przy dobrym oświetleniu i obrocie zdjęcia poprawnie wykrywa twarz i modyfikuje ją. Problemy obejmują niekrytyczne artefakty – białe (czasami czarne) „plamy” na końcowych obrazach. W Monet2Photo problem ten prawie zniknął pod koniec treningu.

*Wyniki AniCycleGAN* : Wyniki AniCycleGAN 1\_1

Wyniki AniCycleGAN 1\_2

*Gdy rozmiar twarzy jest mały, następuje zwykły transfer stylu, który również można zastosować* : Wyniki AniCycleGAN 1\_3

*Sieć może być również używana jako SuperResolution dla twarzy w stylu japońskiej animacji* : Wyniki AniCycleGAN 2\_1

Wyniki AniCycleGAN 2\_2

# <a name="обзор-gui"></a>**Przegląd interfejsu graficznego**
Podsumowując, chcę pokazać, co zaowocowało całą wykonaną pracą. Stworzono prosty, ale dość funkcjonalny GUI do przetwarzania i dalszego zapisywania obrazu. Aplikacja zawiera także krótkie instrukcje (przycisk „?”)

graficzny interfejs użytkownika
