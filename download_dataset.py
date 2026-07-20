from pathlib import Path
import shutil
import kagglehub

dataset_path = Path(
    kagglehub.dataset_download("uciml/sms-spam-collection-dataset")
)

destination = Path("data/raw")
destination.mkdir(parents=True, exist_ok=True)

for file in dataset_path.iterdir():
    shutil.copy(file, destination / file.name)

print("Dataset başarıyla data/raw klasörüne kopyalandı.")