"""
Пересчёт всех экспериментов: последовательное выполнение ноутбуков (код эксперимента — в .ipynb).
Запуск из каталога lab2: python run_experiments.py

Требуется: jupyter / nbconvert, ipykernel.
"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
_nbdir = _root / "notebooks"

NOTEBOOKS = [
    "base_checks.ipynb",
    "experiment_2_2.ipynb",
    "experiment_2_3.ipynb",
    "experiment_2_4.ipynb",
    "experiment_2_5.ipynb",
    "experiment_2_6.ipynb",
    "experiment_track1.ipynb",
]


def main():
    try:
        import nbformat
        from nbconvert.preprocessors import ExecutePreprocessor
    except ImportError as e:
        print("Установите зависимости: pip install jupyter nbconvert", file=sys.stderr)
        raise e

    ep = ExecutePreprocessor(timeout=1800, kernel_name="python3")
    for name in NOTEBOOKS:
        path = _nbdir / name
        if not path.is_file():
            print("Пропуск (нет файла):", path)
            continue
        print("Выполняется:", name, "...")
        with open(path, encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)
        ep.preprocess(nb, {"metadata": {"path": str(_nbdir)}})
        with open(path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        print("  готово:", name)
    print("Все ноутбуки выполнены; графики в report/figures/")


if __name__ == "__main__":
    main()
