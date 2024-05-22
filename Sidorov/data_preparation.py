import pandas as pd
import argparse
from pathlib import Path
import yaml
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from tabulate import tabulate

def parser_args_for_sac():
    parser = argparse.ArgumentParser(description='Paths parser')
    parser.add_argument('--input_dir', '-id', type=str, default='data/raw/',
                        required=False, help='path to input data directory')
    parser.add_argument('--output_dir', '-od', type=str, default='data/prepared/',
                        required=False, help='path to save prepared data')
    parser.add_argument('--params', '-p', type=str, default='params.yaml', required=False,
                        help='file with dvc stage params')
    return parser.parse_args()

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df.drop("anime_id", axis=1, inplace=True)  # Удаляем столбец

    df = df.dropna() # удаление строк с пустыми значениями
    df = df.reset_index(drop=True) # обновление индексации

    # Столбец названия........................

    df['name_len'] = df['name'].str.len()  # Находим длину каждого названия и создаем столбец

    # Функция для разделения строки на слова
    def split_into_words(text):
        letters = list(text)  # Преобразуем строку в массив букв
        current_word = ''  # Инициализируем пустую строку для текущего слова
        words = []  # Инициализируем список для хранения всех слов
        # Перебираем буквы в массиве
        for letter in letters:
            # Если буква не пробел, добавляем ее к текущему слову
            if (letter != ' ') & (letter != ' - ') & (letter != '@') & (letter != '"') & (letter != ';') & (
                    letter != ':') & (letter != '!') & (letter != '?') & (letter != '(') & (letter != ')') & (
                    letter != '[') & (letter != ']') & (letter != '.'):
                current_word += letter
            # Если буква пробел и текущее слово не пустое, добавляем его в список слов и сбрасываем текущее слово
            elif current_word:
                words.append(current_word.lower())  # Приводим слово к нижнему регистру перед добавлением в список
                current_word = ''
        # Если после последней буквы есть непустое текущее слово, добавляем его в список слов
        if current_word:
            words.append(current_word.lower())  # Приводим слово к нижнему регистру перед добавлением в список
        return words
    # Создаем новый столбец, содержащий список слов для каждой строки
    df['words_name'] = df['name'].apply(split_into_words)
    # Применяем функцию к столбцу DataFrame и объединяем списки слов в один список
    all_words = sum(df['words_name'], [])
    # Приводим все слова к нижнему регистру перед подсчетом частоты встречаемости
    all_words_lower = [word.lower() for word in all_words]
    # Считаем частоту встречаемости каждого слова
    word_counts = Counter(all_words_lower)
    # Выбираем наиболее часто встречаемые слова
    top_words = word_counts.most_common(100)
    # Создаем DataFrame из наиболее часто встречаемых слов
    top_words_df = pd.DataFrame(top_words, columns=['unique_words', 'encounter'])
    # Закодируем 1 - есть слова в названии из топ - 100, 0 - слов нет.
    name_words_cod = []
    count1 = oldcount1 = 0
    for i in range(len(df.words_name)):
        for j in range(len(df.words_name[i])):
            for k in range(len(top_words_df.unique_words)):
                if top_words_df.unique_words[k] == df.words_name[i][j]:
                    name_words_cod.append(1)
                    count1 = count1 + 1
                    break
            if count1 != oldcount1:
                break
        if count1 == oldcount1:
            name_words_cod.append(0)
        oldcount1 = count1
    df['name_words_cod'] = name_words_cod
    # Удаляем столбцы
    df = df.drop(['name', 'words_name'], axis=1)

    # Столбец жанров..............................

    # Разделяем строки в столбце 'genre' на отдельные слова, образуем массивы слов
    df['genre'] = df['genre'].str.split(', ')
    # Объединяем все списки из столбца 'genre' в один список
    all_words = [word for sublist in df['genre'] for word in sublist]
    # Находим уникальные жанры
    unique_words = list(set(all_words))
    genre_rating = []
    for i in range(len(unique_words)):
        sum1 = 0.0
        count = 0
        for k in range(len(df.genre)):
            for j in range(len(df.genre[k])):
                if unique_words[i] == df.genre[k][j]:
                    sum1 = sum1 + df.rating[k]
                    count = count + 1
                    break
        sum1 = sum1 / count
        genre_rating.append(sum1)
    genres = pd.DataFrame({'genre': unique_words, 'genre_rating': genre_rating})
    # Сортируем жанры по возрастанию рейтинга
    genres_sorted = genres.sort_values(by='genre_rating')
    genres_sorted = genres_sorted.reset_index(drop=True)  # изменение индексации жанров по возрастанию рейтинга

    # Распределение по комбинациям жанров
    genre_rating = []
    sum1 = sum2 = sum3 = sum4 = sum5 = sum6 = sum7 = sum8 = sum9 = 0.0
    count1 = count2 = count3 = count4 = count5 = count6 = count7 = count8 = count9 = 0
    for i in range(len(df.genre)):
        Romance = Drama = Action = Fantasy = Psychological = Thriller = Sci_Fi = Mecha =  Adventure = Supernatural = 0
        Comedy = School = Harem = Ecchi = Mystery = Detective = 0
        for j in range(len(df.genre[i])):
            if df.genre[i][j] == 'Romance':
                Romance = 1
            if df.genre[i][j] == 'Drama':
                Drama = 1
            if df.genre[i][j] == 'Action':
                Action = 1
            if df.genre[i][j] == 'Fantasy':
                Fantasy = 1
            if df.genre[i][j] == 'Psychological':
                Psychological = 1
            if df.genre[i][j] == 'Thriller':
                Thriller = 1
            if df.genre[i][j] == 'Sci-Fi':
                Sci_Fi = 1
            if df.genre[i][j] == 'Mecha':
                Mecha = 1
            if df.genre[i][j] == 'Adventure':
                Adventure = 1
            if df.genre[i][j] == 'Supernatural':
                Supernatural = 1
            if df.genre[i][j] == 'Comedy':
                Comedy = 1
            if df.genre[i][j] == 'School':
                School = 1
            if df.genre[i][j] == 'Harem':
                Harem = 1
            if df.genre[i][j] == 'Ecchi':
                Ecchi = 1
            if df.genre[i][j] == 'Mystery':
                Mystery = 1
            if df.genre[i][j] == 'Detective':
                Detective = 1
        if Romance & Drama:
            sum1 = sum1 + df.rating[i]
            count1 = count1 + 1
        elif Action & Fantasy:
            sum2 = sum2 + df.rating[i]
            count2 = count2 + 1
        elif Psychological & Thriller:
            sum3 = sum3 + df.rating[i]
            count3 = count3 + 1
        elif Sci_Fi & Mecha:
            sum4 = sum4 + df.rating[i]
            count4 = count4 + 1
        elif Adventure & Supernatural:
            sum5 = sum5 + df.rating[i]
            count5 = count5 + 1
        elif Comedy & Romance & School:
            sum6 = sum6 + df.rating[i]
            count6 = count6 + 1
        elif Harem & Ecchi:
            sum7 = sum7 + df.rating[i]
            count7 = count7 + 1
        elif Mystery & Supernatural:
            sum8 = sum8 + df.rating[i]
            count8 = count8 + 1
        else:
            sum9 = sum9 + df.rating[i]
            count9 = count9 + 1
    sum1 = sum1 / count1
    sum2 = sum2 / count2
    sum3 = sum3 / count3
    sum4 = sum4 / count4
    sum5 = sum5 / count5
    sum6 = sum6 / count6
    sum7 = sum7 / count7
    sum8 = sum8 / count8
    sum9 = sum9 / count9
    genre_rating.append(sum1)
    genre_rating.append(sum2)
    genre_rating.append(sum3)
    genre_rating.append(sum4)
    genre_rating.append(sum5)
    genre_rating.append(sum6)
    genre_rating.append(sum7)
    genre_rating.append(sum8)
    genre_rating.append(sum9)
    genres_rating = pd.DataFrame({'genres': ['Romance&Drama', 'Action&Fantasy', 'Psychological&Thriller',
                                             'Sci-Fi&Mecha', 'Adventure&Supernatural',
                                             'Comedy&Romance&School', 'Harem&Ecchi', 'Mystery&Supernatural',
                                             'Остальные'],
                                  'genres_rating': genre_rating})
    genres_rating_sorted = genres_rating.sort_values(by='genres_rating')
    genres_rating_sorted = genres_rating_sorted.reset_index(drop=True)  # изменение индексации

    # Кодирование по комбинациям жанров
    genre_cod = []
    for i in range(len(df.genre)):
        Romance = Drama = Action = Fantasy = Psychological = Thriller = Sci_Fi = Mecha = Adventure = Supernatural = 0
        Comedy = School = Harem = Ecchi = Mystery = Detective = 0
        for j in range(len(df.genre[i])):
            if df.genre[i][j] == 'Romance':
                Romance = 1
            if df.genre[i][j] == 'Drama':
                Drama = 1
            if df.genre[i][j] == 'Action':
                Action = 1
            if df.genre[i][j] == 'Fantasy':
                Fantasy = 1
            if df.genre[i][j] == 'Psychological':
                Psychological = 1
            if df.genre[i][j] == 'Thriller':
                Thriller = 1
            if df.genre[i][j] == 'Sci-Fi':
                Sci_Fi = 1
            if df.genre[i][j] == 'Mecha':
                Mecha = 1
            if df.genre[i][j] == 'Adventure':
                Adventure = 1
            if df.genre[i][j] == 'Supernatural':
                Supernatural = 1
            if df.genre[i][j] == 'Comedy':
                Comedy = 1
            if df.genre[i][j] == 'School':
                School = 1
            if df.genre[i][j] == 'Harem':
                Harem = 1
            if df.genre[i][j] == 'Ecchi':
                Ecchi = 1
            if df.genre[i][j] == 'Mystery':
                Mystery = 1
            if df.genre[i][j] == 'Detective':
                Detective = 1
        if Romance & Drama:
            for j in range(len(genres_rating_sorted.genres)):
                if 'Romance&Drama' == genres_rating_sorted.genres[j]:
                    genre_cod.append(j)
                    break
        elif Action & Fantasy:
            for j in range(len(genres_rating_sorted.genres)):
                if 'Action&Fantasy' == genres_rating_sorted.genres[j]:
                    genre_cod.append(j)
        elif Psychological & Thriller:
            for j in range(len(genres_rating_sorted.genres)):
                if 'Psychological&Thriller' == genres_rating_sorted.genres[j]:
                    genre_cod.append(j)
        elif Sci_Fi & Mecha:
            for j in range(len(genres_rating_sorted.genres)):
                if 'Sci-Fi&Mecha' == genres_rating_sorted.genres[j]:
                    genre_cod.append(j)
        elif Adventure & Supernatural:
            for j in range(len(genres_rating_sorted.genres)):
                if 'Adventure&Supernatural' == genres_rating_sorted.genres[j]:
                    genre_cod.append(j)
        elif Comedy & Romance & School:
            for j in range(len(genres_rating_sorted.genres)):
                if 'Comedy&Romance&School' == genres_rating_sorted.genres[j]:
                    genre_cod.append(j)
        elif Harem & Ecchi:
            for j in range(len(genres_rating_sorted.genres)):
                if 'Harem&Ecchi' == genres_rating_sorted.genres[j]:
                    genre_cod.append(j)
        elif Mystery & Supernatural:
            for j in range(len(genres_rating_sorted.genres)):
                if 'Mystery&Supernatural' == genres_rating_sorted.genres[j]:
                    genre_cod.append(j)
        else:
            for j in range(len(genres_rating_sorted.genres)):
                if 'Остальные' == genres_rating_sorted.genres[j]:
                    genre_cod.append(j)
    df['genre_cod'] = genre_cod

    # Кодирование жанров по их индексу
    for i in range(len(df.genre)):
        genre_code = []
        for k in range(len(df.genre[i])):
            for j in range(len(genres_sorted.genre)):
                if df.genre[i][k] == genres_sorted.genre[j]:
                    genre_code.append(j)
                    break
        df.loc[i, "genre"] = [pd.Series(numbers) for numbers in genre_code]
    # Создание столбца со средним значением
    df['sr_genre'] = df['genre'].apply(lambda x: sum(x) / len(x))
    # Удаляем столбец
    df = df.drop(['genre'], axis=1)

    # Столбец типа.................................

    # Находим уникальные типы аниме
    unique_type = list(set(df['type']))
    # Определяем средний рейтинг по всем типам аниме
    type_rating = []
    for i in range(len(unique_type)):
        sum1 = 0.0
        count = 0
        for k in range(len(df.type)):
            if unique_type[i] == df.type[k]:
                sum1 = sum1 + df.rating[k]
                count = count + 1
        sum1 = sum1 / count
        type_rating.append(sum1)
    types = pd.DataFrame({'type': unique_type, 'type_rating': type_rating})
    # Сортируем типы аниме по возрастанию рейтинга
    types_sorted = types.sort_values(by='type_rating')
    # Закодируем type
    types_sorted = types_sorted.reset_index(drop=True)  # изменение индексации
    type_code = []
    for i in range(len(df.type)):
        for j in range(len(types_sorted.type)):
            if df.type[i] == types_sorted.type[j]:
                type_code.append(j)
                break
    df['type_code'] = type_code
    # Удаляем столбец
    df = df.drop(['type'], axis=1)

    # Столбец кол-ва эпизодов.................................

    # Удаление строк, в которых в столбце 'episodes' встречается 'Unknown'
    df = df.query('episodes != "Unknown"')
    df = df.reset_index(drop=True)  # обновление индексации
    df['episodes'] = df['episodes'].astype(int)  # преобразование столбца в int

    return df


if __name__ == '__main__':
    args = parser_args_for_sac()
    with open(args.params, 'r') as f:
        params_all = yaml.safe_load(f)
    params = params_all['data_preparation']

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(exist_ok=True, parents=True)

    for data_file in input_dir.glob('*.csv'):
        full_data = pd.read_csv(data_file)
        cleaned_data = clean_data(df=full_data)
        X, y = cleaned_data.drop("rating", axis=1), cleaned_data['rating']
        X_train , X_test, y_train, y_test = train_test_split(X, y,
                                                            train_size=params.get('train_test_ratio'),
                                                            random_state=params.get('random_state'))
        X_train , X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                          train_size=params.get('train_val_raitio'),
                                                          random_state=params.get('random_state'))
        X_full_name = output_dir / 'X_full.csv'
        y_full_name = output_dir / 'y_full.csv'
        X_train_name = output_dir / 'X_train.csv'
        y_train_name = output_dir / 'y_train.csv'
        X_test_name = output_dir / 'X_test.csv'
        y_test_name = output_dir / 'y_test.csv'
        X_val_name = output_dir / 'X_val.csv'
        y_val_name = output_dir / 'y_val.csv'

        X.to_csv(X_full_name, index=False)
        y.to_csv(y_full_name, index=False)
        X_train.to_csv(X_train_name, index=False)
        y_train.to_csv(y_train_name, index=False)
        X_test.to_csv(X_test_name, index=False)
        y_test.to_csv(y_test_name, index=False)
        X_val.to_csv(X_val_name, index=False)
        y_val.to_csv(y_val_name, index=False)