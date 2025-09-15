import pandas
import folium

data = pandas.read_csv("1_w.csv", sep=';', encoding='cp1251')

#print(data.head(3))

data['xy'] = data.apply(lambda x: [x['Широта'], x['Долгота']], axis = 1)

#print(data.head(3))

m = folium.Map(data.loc[0,'xy'], zoom_start=13)

route = folium.PolyLine(list(data['xy']), вес = 3, color = 'green', непрозрачность = 0.8).add_to(m)

# for i in ind:
    # if '30' in data.loc[i, 'point']:
        # folium.CircleMarker(location=data.loc[i, 'xy'], radius=2, popup=str(i)+"_"+str(data))

display(m)



import pandas as pd
import os
import glob

def merge_csv_files_basic(folder_path, output_file='merged_data.csv'):

    # Находим все CSV файлы в папке
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    
    if not csv_files:
        print("CSV файлы не найдены!")
        return None
    
    print(f"Найдено {len(csv_files)} CSV файлов")
    
    # Читаем и объединяем все файлы
    dataframes = []
    for file in csv_files:
        try:
            df = pd.read_csv(file, sep=';', encoding='cp1251')
            df['source_file'] = os.path.basename(file)  # Добавляем имя файла
            dataframes.append(df)
            print(f"Обработан: {file} - {len(df)} строк")
        except Exception as e:
            print(f"Ошибка при чтении {file}: {e}")
    
    # Объединяем все DataFrame
    merged_df = pd.concat(dataframes, ignore_index=True)
    
    # Сохраняем результат
    merged_df.to_csv(output_file, index=False)
    print(f"Объединенный файл сохранен как: {output_file}")
    print(f"Итоговый размер: {len(merged_df)} строк, {len(merged_df.columns)} столбцов")
    
    return merged_df

df = merge_csv_files_basic('route/', output_file='full_dataset.csv')










import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def split_dataset_basic(merged_df, target_column, test_size=0.2, random_state=42):
    
    print("=== РАЗДЕЛЕНИЕ ДАТАСЕТА НА TRAIN/TEST ===")
    
    # Проверяем наличие целевой колонки
    if target_column not in merged_df.columns:
        raise ValueError(f"Целевая колонка '{target_column}' не найдена в датасете")
    
    # Отделяем признаки и целевую переменную
    X = merged_df.drop(columns=[target_column])
    y = merged_df[target_column]
    
    print(f"Размер всего датасета: {len(merged_df)} строк")
    print(f"Количество признаков: {X.shape[1]}")
    print(f"Целевая переменная: {target_column}")
    print(f"Уникальные значения целевой переменной: {y.nunique()}")
    print(f"Распределение целевой переменной:\n{y.value_counts()}")
    
    # Разделяем на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # Сохраняем распределение классов
    )
    
    print(f"\nРазмер тренировочной выборки: {len(X_train)} строк ({1-test_size:.0%})")
    print(f"Размер тестовой выборки: {len(X_test)} строк ({test_size:.0%})")
    
    # Проверяем распределение классов
    print(f"\nРаспределение в тренировочной выборке:")
    print(y_train.value_counts())
    print(f"\nРаспределение в тестовой выборке:")
    print(y_test.value_counts())
    
    return X_train, X_test, y_train, y_test

print("=== ВЫДЕЛЕНИЕ В ДАТАСЕТЕ ДАННЫХ АКСЕЛЕРОМЕТРА И КАТЕГОРИИ ЯМЫ ===")

important_columns = ['Ускорение по оси X', 'Ускорение по оси Y', 'Ускорение по оси Z', 'nom_hole', 'point']
selected_df = df[important_columns].copy()

print(f"Исходный размер: {df.shape}")
print(f"После фильтрации: {selected_df.shape}")
print(f"Оставшиеся колонки: {list(selected_df.columns)}")


X_train, X_test, y_train, y_test = split_dataset_basic(merged_df = selected_df, target_column = 'point', test_size=0.3, random_state=42)