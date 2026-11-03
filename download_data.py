"""Download Kaggle dataset version 1; do not commit raw data."""
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

URL = ('https://www.kaggle.com/api/v1/datasets/download/'
       'fedesoriano/heart-failure-prediction?datasetVersionNumber=1')

def main():
    target = Path(__file__).resolve().parent / 'data' / 'heart.csv'
    if target.exists():
        print('data/heart.csv already exists; no changes made.')
        return
    try:
        with urlopen(URL, timeout=60) as response:
            archive = response.read()
        with ZipFile(BytesIO(archive)) as archive_file:
            content = archive_file.read('heart.csv')
    except Exception as error:
        raise SystemExit('Automatic download failed. Download version 1 manually from '
                         'https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction '
                         'and place heart.csv in data/.') from error
    target.parent.mkdir(exist_ok=True)
    target.write_bytes(content)
    print('Downloaded data/heart.csv (dataset version 1).')

if __name__ == '__main__':
    main()
