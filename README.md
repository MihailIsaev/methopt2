# Лабораторная работа 2

Пакет №3 (Pseudo-Huber + Smoothed SVM), функция Била, трек 1 (Steihaug–Toint и модификации гессиана).

Ключевое ограничение этой лабы: на ML-задачах не используется явно построенный гессиан. Реализовано только произведение `H(x) v` через `hess_vec` без сборки матрицы в памяти.

## Структура

- `README.md` — этот файл.
- `requirements.txt` — зависимости.
- `src/` — код: `optimization.py`, `oracles.py`, `utils.py`, `ml_tools.py`, `experiments_common.py`.
- `notebooks/` — `base_checks.ipynb`, `experiment_2_2.ipynb` … `experiment_2_6.ipynb`, `experiment_track1.ipynb`.
- `report/` — итоговый отчёт и **отдельный markdown на каждый эксперимент**; графики в `report/figures/` (`.pdf` и `.png`).
- `data/` — локальные копии LIBSVM (например `a1a`; при отсутствии подкачивается с сайта C.J. Lin). Для `triazines_scale` сначала ищется копия из `MetOpt_lab1/lab1/data/libsvm/`, затем локальная `lab2/data/triazines_scale`.
- `run_experiments.py` — последовательное выполнение ноутбуков через nbconvert (`python run_experiments.py` из каталога `lab2`).
- `scripts/write_experiment_notebooks.py` — скрипт генерации ноутбуков экспериментов.

Регрессия по умолчанию: `MetOpt_lab1/lab1/data/libsvm/triazines_scale`. Классификация: `data/a1a`.
