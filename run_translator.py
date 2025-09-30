import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import re
import logging
from tqdm import tqdm
import itertools

# --- НАСТРОЙКИ ---
MODEL_NAME = "Helsinki-NLP/opus-mt-ru-en"
INPUT_FILE = "input.csv"
OUTPUT_FILE = "output.csv"

INPUT_ENCODING = "cp1251"
OUTPUT_ENCODING = "utf-8"
CSV_DELIMITER = ";" # Используется только для записи

# --- ПАРАМЕТРЫ ПРОИЗВОДИТЕЛЬНОСТИ ---
# CHUNK_SIZE - сколько строк читаем в память за раз.
CHUNK_SIZE = 5500

# BATCH_SIZE - сколько фраз модель обрабатывает на GPU параллельно.
# УМЕНЬШЕНИЕ этого значения снижает нагрузку на GPU, но может замедлить процесс.
BATCH_SIZE = 512

LOG_FILE = "translation_final.log"
LOG_LEVEL = "INFO"
USE_GPU = True
# --- КОНЕЦ НАСТРОЕК ---

# Настройка логирования
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8', mode='w'),
        logging.StreamHandler()
    ]
)

def load_model():
    """Загружает модель и токенизатор, определяя устройство (GPU/CPU)."""
    device = 0 if torch.cuda.is_available() and USE_GPU else -1
    logging.info(f"Using device: {'GPU' if device == 0 else 'CPU'}")
    if device == 0:
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        
        translator = pipeline(
            "translation",
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=BATCH_SIZE
        )
        logging.info("Translation model loaded successfully.")
        return translator
    except Exception as e:
        logging.error(f"Failed to load model. Error: {e}")
        raise

# Паттерн для поиска кириллицы. Находит отдельные слова и фразы.
RUSSIAN_PATTERN = re.compile(r'[а-яА-ЯёЁ]+(?:[_\s-][а-яА-ЯёЁ]+)*')

def process_chunk(df: pd.DataFrame, translator):
    """
    Обрабатывает чанк (порцию) данных: находит, переводит и заменяет русский текст.
    """
    target_column = 'content' # Теперь у нас только одна колонка с сырым текстом
    text_series = df[target_column].astype(str)

    # 1. Сбор всех уникальных русских фраз со всего чанка
    phrases_to_translate = set()
    for text in text_series:
        found_phrases = RUSSIAN_PATTERN.findall(text)
        phrases_to_translate.update(found_phrases)

    if not phrases_to_translate:
        logging.info("No Russian phrases found in this chunk. Skipping translation.")
        return df

    # 2. Пакетный перевод (один вызов для всего чанка)
    try:
        unique_phrases_list = list(phrases_to_translate)
        logging.info(f"Translating {len(unique_phrases_list)} unique phrases...")
        
        translated_results = translator(unique_phrases_list, max_length=128, num_beams=5, early_stopping=True, no_repeat_ngram_size=2)
        
        translation_map = {
            original: result['translation_text'] 
            for original, result in zip(unique_phrases_list, translated_results)
        }
        logging.info("Batch translation successful.")
    except Exception as e:
        # Пишем в лог-файл, как и раньше
        logging.error(f"CRITICAL ERROR during batch translation: {e}")
        
        # А теперь выводим сообщение прямо в консоль
        # tqdm.write гарантирует, что сообщение не затрется прогресс-баром
        from tqdm import tqdm
        tqdm.write("\n" + "="*80)
        tqdm.write(f"!!! КРИТИЧЕСКАЯ ОШИБКА: НЕХВАТКА ВИДЕОПАМЯТИ ПРИ ОБРАБОТКЕ ЧАНКА !!!")
        tqdm.write(f"!!! ДЕТАЛИ: {e} !!!")
        tqdm.write(f"!!! ЭТОТ ЧАНК БУДЕТ ПРОПУЩЕН (останется на русском). УМЕНЬШИТЕ CHUNK_SIZE. !!!")
        tqdm.write("="*80 + "\n")
        
        # Возвращаем оригинальный чанк, чтобы скрипт не падал
        return df

    # 3. Эффективная замена с помощью re.sub
    def replace_russian_text(field: str) -> str:
        if not isinstance(field, str) or not translation_map:
            return field
            
        def replace_callback(match):
            phrase = match.group(0)
            return translation_map.get(phrase, phrase)
        
        return RUSSIAN_PATTERN.sub(replace_callback, field)

    # 4. Применяем функцию замены ко всей колонке
    df[target_column] = text_series.apply(replace_russian_text)
    
    return df


# v.1.1 Функция, проверяет output и начинает с того места, где остановилась

def main():
    """Главная функция для запуска процесса перевода с возможностью возобновления."""
    logging.info("Starting CSV translation process.")

    # --- Логика возобновления ---
    processed_lines_count = 0
    try:
        # Проверяем, есть ли уже выходной файл и считаем в нем строки
        with open(OUTPUT_FILE, 'r', encoding=OUTPUT_ENCODING) as f_out_check:
            processed_lines_count = sum(1 for row in f_out_check)

        if processed_lines_count > 0:
            print(f"Найден существующий выходной файл с {processed_lines_count} строками.")
            print(f"Возобновляем работу с {processed_lines_count + 1}-й строки...")
            logging.info(f"Resuming from line {processed_lines_count + 1}.")

    except FileNotFoundError:
        # Если файла нет, начинаем с нуля
        print("Выходной файл не найден. Начинаем с самого начала.")
        logging.info("Output file not found. Starting from scratch.")
        # Создаем пустой файл, чтобы скрипт мог в него дописывать
        with open(OUTPUT_FILE, 'w', encoding=OUTPUT_ENCODING) as f:
            pass
    # --- Конец логики возобновления ---

    translator = load_model()

    try:
        # Считаем общее количество строк в исходном файле для прогресс-бара
        total_lines = sum(1 for row in open(INPUT_FILE, 'r', encoding=INPUT_ENCODING))
        logging.info(f"Found {total_lines} total lines in {INPUT_FILE}.")
    except Exception as e:
        logging.error(f"Could not count lines in input file: {e}. Progress bar might be inaccurate.")
        total_lines = None

    try:
        with open(INPUT_FILE, 'r', encoding=INPUT_ENCODING) as f_in:

            # --- Пропускаем уже обработанные строки ---
            if processed_lines_count > 0:
                print("Пропускаем обработанные строки в исходном файле...")
                # Эффективно "проматываем" файл до нужной позиции
                for _ in range(processed_lines_count):
                    next(f_in)

            # Устанавливаем начальное значение для прогресс-бара
            pbar = tqdm(initial=processed_lines_count, total=total_lines, desc="Translating file", unit="lines")

            while True:
                lines_iterator = itertools.islice(f_in, CHUNK_SIZE)
                lines = [line.strip() for line in lines_iterator]

                if not lines:
                    break

                chunk_df = pd.DataFrame(lines, columns=['content'])
                processed_chunk = process_chunk(chunk_df, translator)

                processed_chunk.to_csv(
                    OUTPUT_FILE,
                    mode='a',          # 'a' - дописать в конец файла
                    header=False,      # Не записывать заголовок колонки
                    index=False,       # Не записывать индекс pandas (0, 1, 2...)
                    encoding=OUTPUT_ENCODING
                )

                pbar.update(len(chunk_df))

            pbar.close()

        logging.info("Translation completed successfully.")
        print(f"\n✅ Перевод завершен! Результат в файле: {OUTPUT_FILE}")

    except Exception as e:
        logging.critical(f"A critical error occurred: {e}", exc_info=True)
        print(f"\n❌ Критическая ошибка! Проверьте лог-файл: {LOG_FILE}")


if __name__ == '__main__':
    main()