import numpy as np
import nibabel as nib
import os
import glob

# Директория с исходными .npz файлами
input_dir = "/home/alexandra/test-folder/SDF4CHD-doubleM/data/gdrive_downloads"
# Директория для сохранения .nii.gz файлов
output_dir = "/home/alexandra/test-folder/SDF4CHD-doubleM/data/artery_nii.gz_files/"

# Ключ, под которым хранятся данные изображения в .npz файлах
data_key = 'artery'

# Убедимся, что выходная директория существует
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Создана директория: {output_dir}")

# Ищем все .npz файлы в исходной директории
# Пожалуйста, еще раз убедитесь, что input_dir указан правильно и файлы .npz действительно там.
# Проверить можно командой: ls /home/alexandra/test-folder/SDF4CHD-doubleM/data/gdrive_downloads/artery_npz_files/
npz_files = glob.glob(os.path.join(input_dir, "*.npz"))

if not npz_files:
    print(f"В директории {input_dir} не найдены .npz файлы. Пожалуйста, проверьте путь input_dir и наличие файлов.")
else:
    print(f"Найдено {len(npz_files)} .npz файлов для конвертации.")

for npz_file_path in npz_files:
    try:
        print(f"Обработка файла: {npz_file_path}")

        # Загружаем .npz файл
        data_npz = np.load(npz_file_path)

        # Извлекаем массив данных изображения
        image_data = data_npz[data_key]
        
        # Закрываем .npz файл после извлечения данных
        data_npz.close()

        # Создаем единичную аффинную матрицу
        affine_matrix = np.eye(4) 

        # Создаем Nifti1Image объект
        nifti_image = nib.Nifti1Image(image_data, affine_matrix)

        # Формируем имя выходного файла
        base_filename = os.path.basename(npz_file_path)
        nifti_filename = base_filename.replace(".npz", ".nii.gz")
        output_file_path = os.path.join(output_dir, nifti_filename)

        # Сохраняем Nifti файл
        nib.save(nifti_image, output_file_path)
        print(f"Сохранен файл: {output_file_path}")

    except Exception as e:
        print(f"Ошибка при обработке файла {npz_file_path}: {e}")

print("Конвертация завершена.")
