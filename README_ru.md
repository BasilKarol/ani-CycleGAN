<p align="center">
	<img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54"/>
	<img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white"/>
</p>

# AniCycleGAN - Финальный проект из курса Deep Learning School Василия Короля

### Проект по генерации и изменению стиля изображений при помощи CycleGAN архитектуры

<p align="center">
	<img src="./images/ACGAN_preview_scaled.jpg" />
</p>

## Оглавление:
* [Техническая информация](#тех-инф)
* [Использование](#запуск)
* [Результаты проделанной работы](#результаты)
* [Обзор GUI](#результаты)

<a name="тех-инф"/>

## Техническая информация

Для создания проекта были использованны:
* python 3.10
* jupyter notebook (kaggle GPU)

Все библиотеки описаны в файле **REQUIREMENTS.txt**

<a name="запуск"/>

## Использование

Для запуска  GUI программы достаточно запустить файл main в директории проекта при помощи python (windows 10):
```
your\folder\with\main\file> python main.py
```
<a name="результаты"/>

## Результаты проделанной работы

За основу взята официальная статья "***Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Network 2017***" (https://arxiv.org/pdf/1703.10593.pdf)
Реализация сети началась с задачи Monet2Photo - научить сеть переносить фото в стиль картин Моне / делать картины Моне фотореалистичными. 
Данная работа описана в ноутбуке **cyclegan_demo.ipynb**

### *Результаты cyclegan_demo Моне в Фото*:
![cyclegan_demo1 Results](./images/cyclegan_ex1.jpg)
<p align="center">
	<img src="./images/cyclegan_ex2_new.jpg" />
</p>

### *И наоборот*:
![cyclegan_demo3 Results](./images/cyclegan_ex3_new.jpg)

**Основная поставленная задача** - преобразование реального лица в соответствующие ему лицо в стиле японской анимации с сохранением первоначальных черт.
Основное обучение и архитектура сететй описана в файле **AniCycleGAN.ipynb**

Первоначально сеть тренировалась на датасете Flickr Faces (https://www.kaggle.com/xhlulu/flickrfaceshq-dataset-nvidia-resized-256px). 
Однако из-за разнообразия пола/возраста людей, а так же различия в их мастштабе сеть НЕ смогла правильно переносить стиль. 
После некоторых поисков был найден идеально подходяший под задачу датасет selfie2anime (https://www.kaggle.com/arnaud58/selfie2anime)

Обучение проводилось при помощи выделенной платформой kaggle видеокарты Nvidia Tesla K80. Время полного обучения заняло ~30 часов.
В итоговом варианте в 80% случаев сеть успешно переносить стиль, а так же при хорошем освещении и повороте фотографии правильно детектирует лицо и видоизменяет его.
Из проблем можно выделить некритичные артефакты - белые (иногда черные) 'пятна' на итоговых изображениях. В работе Monet2Photo данная проблема практически пропала к концу обучения

### *Результаты AniCycleGAN*:
![Результаты AniCycleGAN 1_1](./images/anigan_ex1.png)

![Результаты AniCycleGAN 1_2](./images/anigan_ex2.png)

### *При маленьком размере лиц идет обычный перенос стиля, что так же может быть использованно*:
![Результаты AniCycleGAN 1_3](./images/anigan_ex0.png)

### *Так же сеть может использоваться как SuperResolution для лиц стиля японской анимации*:
![Результаты AniCycleGAN 2_1](./images/anigan_ex3.png)

![Результаты AniCycleGAN 2_2](./images/anigan_ex4.png)

<a name="обзор"/>

# Обзор GUI
В заключении хочу показать то, во что вылилась вся проделанная работа. Был создан простой, но достаточно функциональный GUI для обработки и дальнейшего сохранения изображения. 
В приложении также имеется краткая инструкция (кнопка '?')

![GUI](./images/GUI_interface.png)
