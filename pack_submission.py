"""Скрипт сборки ZIP-сабмита для платформы хакатона ПСБ.

Требования регламента:
1. Файлы должны находиться в корне архива (без вложенной папки).
2. Обязательная точка входа: solution.py.
3. requirements.txt со списком зависимостей.
4. README.md с описанием решения.
5. Максимальный размер до 128 МБ.
"""

import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SUBMISSION_ZIP = BASE_DIR / "psb_submission.zip"

FILES_TO_PACK = [
    "solution.py",
    "requirements.txt",
    "README.md",
]


def pack():
    print(f"Сборка сабмита в {SUBMISSION_ZIP}...")
    with zipfile.ZipFile(SUBMISSION_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fname in FILES_TO_PACK:
            fpath = BASE_DIR / fname
            if not fpath.is_file():
                # Если файла нет в корне, копируем или проверяем
                raise FileNotFoundError(f"Файл {fname} не найден в {BASE_DIR}")
            z.write(fpath, arcname=fname)
            print(f"  + {fname} ({fpath.stat().st_size} байт)")

    print(f"\n🎉 Сабмит успешно собран: {SUBMISSION_ZIP}")
    print(f"Итоговый размер архива: {SUBMISSION_ZIP.stat().st_size} байт")


if __name__ == "__main__":
    pack()
