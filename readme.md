# Скрипт для пакетного перевода текста RU→EN в CSV-файлах с помощью AI/нейросети  
*(English description below)*  

🇷🇺 Этот README сначала на русском, ниже — английская версия.  

Этот скрипт предназначен для автоматического перевода русскоязычного текста на английский язык внутри больших CSV-файлов. Он оптимизирован для работы с файлами объемом в сотни мегабайт и миллионы строк, эффективно используя ресурсы графического процессора (GPU) для ускорения процесса.

## Основные возможности

* **Обработка больших файлов:** Скрипт читает исходный файл по частям (чанками), что позволяет обрабатывать файлы, превышающие объем оперативной памяти.
* **Ускорение на GPU:** Использует библиотеку PyTorch и Hugging Face Transformers для выполнения вычислений на видеокарте NVIDIA (CUDA), что многократно ускоряет перевод.
* **Пакетная обработка (Batching):** Для максимальной эффективности GPU, скрипт собирает уникальные фразы и переводит их большими пакетами.
* **Интеллектуальное извлечение текста:** С помощью регулярных выражений находит и переводит только русскоязычные фрагменты, не затрагивая код, разметку и другие символы в строках.
* **Надежная запись результата:** Использует оптимизированные методы для записи данных, предотвращая ошибки нехватки памяти (`MemoryError`) на финальном этапе.
* **Настраиваемость:** Ключевые параметры, такие как размеры чанков и пакетов, вынесены в блок конфигурации для легкой настройки под конкретную систему.

## Требования

Для работы скрипта необходим Python 3.x и следующие библиотеки. Зависимости в requirements.txt. Установить их можно командой:

```bash
pip install -r requirements.txt
```
**Важно:** Для работы на GPU у вас должна быть установлена видеокарта NVIDIA и соответствующая версия CUDA, совместимая с вашей версией PyTorch. Для поддержки CUDA нужно установить torch со следующими параметрами
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Конфигурация

Все основные параметры находятся в начале скрипта в блоке настроек.

* `MODEL_NAME`: Название модели для перевода из репозитория Hugging Face. По умолчанию `Helsinki-NLP/opus-mt-ru-en`.
* `INPUT_FILE`: Имя исходного CSV-файла.
* `OUTPUT_FILE`: Имя файла, в который будет сохранен результат.
* `INPUT_ENCODING` / `OUTPUT_ENCODING`: Кодировки исходного и конечного файлов.
* `CHUNK_SIZE`: **Количество строк**, считываемых из исходного файла за один раз. Это ключевой параметр для балансировки нагрузки на **CPU и оперативную память (RAM)**.
* `BATCH_SIZE`: **Количество фраз**, которые одновременно обрабатываются на **GPU**. Это ключевой параметр для балансировки нагрузки на **видеопамять (VRAM)**.
* `LOG_FILE` / `LOG_LEVEL`: Настройки для файла логирования, куда записываются все шаги и ошибки.
* `USE_GPU`: `True` для использования GPU, `False` для принудительного использования CPU.

## Использование

1.  Настройте параметры в конфигурационном блоке скрипта.
2.  Поместите исходный `input.csv` в ту же папку.
3.  Запустите скрипт из командной строки:
    ```bash
    python run_translator.py
    ```
4.  Процесс выполнения будет отображаться с помощью прогресс-бара. По завершении в указанном `OUTPUT_FILE` появится переведенный файл.

## Как работает код

Процесс работы скрипта можно разбить на несколько ключевых этапов:

**1. Инициализация и загрузка модели (`main` и `load_model`)**
   - При запуске скрипт настраивает систему логирования.
   - Вызывается функция `load_model()`, которая определяет наличие GPU.
   - С помощью библиотеки `transformers` загружается указанная нейросетевая модель (`Helsinki-NLP/opus-mt-ru-en`) и ее токенизатор. Модель перемещается в память видеокарты.
   - Создается объект `pipeline` — это высокоуровневый интерфейс `transformers`, который инкапсулирует всю логику токенизации, перевода и постобработки.

**2. Чтение файла по частям (`main`)**
   - Основной цикл в функции `main` не загружает весь CSV-файл в память.
   - С помощью `itertools.islice` он эффективно считывает из файла порцию строк, равную `CHUNK_SIZE`. Это позволяет работать с файлами любого размера.
   - Каждая такая порция строк преобразуется в `pandas.DataFrame` для удобной дальнейшей обработки.

**3. Обработка одного чанка (`process_chunk`)**
   - Эта функция является ядром всей логики перевода.
   - **Шаг 3a: Извлечение и дедупликация фраз.**
     - Функция проходит по каждой строке в DataFrame.
     - С помощью регулярного выражения `RUSSIAN_PATTERN` она находит все фрагменты, содержащие кириллицу.
     - Все найденные фразы добавляются в объект `set`. `Set` — это структура данных в Python, которая автоматически хранит только **уникальные** значения. Это важнейшая оптимизация, которая предотвращает многократный перевод одного и того же слова (например, слова "ошибка").
   - **Шаг 3b: Пакетный перевод на GPU.**
     - Уникальные фразы из `set` передаются одним большим списком в объект `translator`.
     - Здесь происходит магия: `pipeline` разбивает этот список на мини-пакеты размером `BATCH_SIZE` и отправляет их на GPU. GPU, благодаря своей архитектуре, обрабатывает сотни фраз параллельно, что обеспечивает колоссальное ускорение.
     - При вызове переводчика используются специальные параметры (`max_length`, `no_repeat_ngram_size` и др.), чтобы предотвратить "глюки" модели, такие как зацикливание и генерация бессмысленного текста.
     - Результатом является словарь-карта вида `{'оригинал': 'перевод'}`.
   - **Шаг 3c: Безопасная замена текста.**
     - Для каждой исходной строки в чанке снова используется регулярное выражение `RUSSIAN_PATTERN`, но на этот раз с функцией `re.sub()`.
     - Для каждого найденного русского фрагмента `re.sub` вызывает callback-функцию `replace_callback`, которая смотрит в созданный ранее словарь и подставляет соответствующий перевод.
     - Этот подход гарантирует, что заменяются **только** русские слова, а весь окружающий код, знаки пунктуации и разметка остаются нетронутыми.
   - **Шаг 3d: Возврат результата.**
     - Функция возвращает `DataFrame` с уже переведенным текстом.

**4. Запись результата и повторение (`main`)**
   - Обработанный `DataFrame` записывается в выходной файл с помощью метода `.to_csv()`. Этот метод оптимизирован для работы с большими данными и не вызывает ошибок нехватки памяти, так как записывает данные на диск порциями, а не создает один гигантский объект в памяти.
   - Файл открывается в режиме дозаписи (`mode='a'`), поэтому каждый новый обработанный чанк добавляется в конец файла.
   - Прогресс-бар `tqdm` обновляется, и цикл повторяется для следующего чанка, пока весь исходный файл не будет прочитан.

---

# Script for Batch RU→EN Translation in CSV Files Using AI/Neural Networks  
*(Русское описание выше)*  

🇬🇧 This README is first in Russian, scroll up for details.  

Here’s a clear English translation of your text, adapted to look natural in a GitHub README:

---

# Script for Batch RU→EN Translation in CSV Files Using AI/Neural Networks

This script is designed to automatically translate Russian text into English inside large CSV files. It is optimized to handle files hundreds of megabytes in size and millions of rows, efficiently utilizing GPU resources to speed up the process.

## Key Features

* **Large file processing:** Reads the input file in chunks, allowing translation of files larger than available system memory.
* **GPU acceleration:** Uses PyTorch and Hugging Face Transformers to run computations on an NVIDIA GPU (CUDA), greatly accelerating translation.
* **Batching:** Collects unique phrases and translates them in large batches for maximum GPU efficiency.
* **Smart text extraction:** Uses regular expressions to detect and translate only Russian text fragments, leaving code, markup, and other symbols intact.
* **Robust output writing:** Optimized CSV writing methods prevent `MemoryError` during the final write stage.
* **Configurable:** Core parameters like chunk size and batch size are adjustable in a dedicated configuration block.

## Requirements

Python 3.x is required, along with the dependencies listed in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```

**Important:** To use GPU acceleration, you need an NVIDIA GPU and a compatible CUDA version for PyTorch. For CUDA support, install Torch with:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Configuration

All key parameters are defined at the top of the script:

* `MODEL_NAME`: Hugging Face model name (default: `Helsinki-NLP/opus-mt-ru-en`).
* `INPUT_FILE`: Input CSV file.
* `OUTPUT_FILE`: Output CSV file.
* `INPUT_ENCODING` / `OUTPUT_ENCODING`: File encodings.
* `CHUNK_SIZE`: **Number of rows** read from the input file per iteration. Balances CPU and RAM usage.
* `BATCH_SIZE`: **Number of phrases** processed per GPU batch. Balances GPU VRAM usage.
* `LOG_FILE` / `LOG_LEVEL`: Logging configuration.
* `USE_GPU`: `True` to enable GPU, `False` to force CPU.

## Usage

1. Adjust the configuration block in the script.
2. Place your source `input.csv` in the same directory.
3. Run the script:

   ```bash
   python run_translator.py
   ```
4. A progress bar will track progress. The translated output will be saved to `OUTPUT_FILE`.

## How It Works

**1. Initialization and model loading (`main`, `load_model`)**

* Sets up logging.
* Detects GPU availability.
* Loads the translation model (`Helsinki-NLP/opus-mt-ru-en`) and tokenizer with Hugging Face `transformers`.
* Creates a `pipeline` object to handle tokenization, translation, and post-processing.

**2. Reading the file in chunks (`main`)**

* Iteratively reads the CSV file in chunks (`CHUNK_SIZE`) using `itertools.islice`.
* Each chunk is converted into a `pandas.DataFrame` for processing.

**3. Processing a chunk (`process_chunk`)**

* **Step 3a: Extracting and deduplicating phrases**

  * Uses regex to find Cyrillic fragments in each row.
  * Stores unique phrases in a Python `set` to avoid duplicate translations.
* **Step 3b: Batch translation on GPU**

  * Passes unique phrases to the `translator` pipeline.
  * Processes them in batches (`BATCH_SIZE`) on GPU.
  * Creates a mapping dictionary `{original: translation}`.
* **Step 3c: Safe text replacement**

  * Uses regex substitution with a callback to replace only Russian fragments while preserving surrounding code/markup.
* **Step 3d: Returning the result**

  * Returns a translated `DataFrame`.

**4. Writing results (`main`)**

* Saves translated data to CSV using `.to_csv()` in append mode.
* Writes data in streaming mode to avoid memory issues.
* Updates progress bar until the entire input file is processed.

---
