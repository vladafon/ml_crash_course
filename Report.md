Работу выполнили студенты группы М8О-214М-21  
Иоффе Владислав  
Сайфудинов Роман  
Богуславский Андрей  

В работе использована RNN нейросейть на основе LSTM (tensorflow keras)  
_________________________________________________________________
Model: "sequential"  
| Layer (type)          |      Output Shape          |    Param     |
|--|--|--|
| text_vectorization_1 |(TextVectorization) (None, None)       |      0    |    
|embedding (Embedding)      | (None, None, 32)   |       448000    |
|bidirectional (Bidirectional) | (None, 64)         |      16640    |     
|dense (Dense)         |      (None, 32)          |      2080      |
|dense_1 (Dense)      |       (None, 1)    |             33        |
                                                                 
  
Total params: 466,753  
Trainable params: 466,753  
Non-trainable params: 0  
_________________________________________________________________

В результате подбора параметров удалось достичь точности в 0.88. 
Был произведен анализ датасета, в ходе которого было установлено, что количество уникальных слов в тестах 14000+, средняя длина предложений 35 слов.
Поэтому в качестве параметров был выбран размер словаря 14000 и размер embeddingа в 32.

Перед использованием, полученные тексты очищались от спецсимволов и цифр, приводились к нижнему регистру, убирались стоп слова. 

Посмотреть, как производился анализ датасета можно в файле https://github.com/vladafon/ml_crash_course/blob/main/test_notebook/test.ipynb
