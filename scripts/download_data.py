import os
import requests


def download_data():
    file_names = [
        *[f'fold_{i}_data.txt' for i in range(5)],
        *[f'fold_frontal_{i}_data.txt' for i in range(5)],
        'aligned.tar.gz',
        'faces.tar.gz'
    ]
    block_size = 1024  # 1 Kilobyte

    base_url = 'http://www.cslab.openu.ac.il/download/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification/'
    data_path = os.path.join(os.getcwd(), 'data')

    if not os.path.exists(data_path):
        os.mkdir(data_path)

    username = 'adiencedb'
    password = 'adience'

    for file_name in file_names:
        print(f'Downloading {file_name}...')

        file_url = base_url + file_name
        file_path = os.path.join(data_path, file_name)
        r = requests.get(file_url, auth=(username, password))

        if r.status_code == 200:
            with open(file_path, 'wb') as out:
                for bits in r.iter_content():
                    out.write(bits)


if __name__ == '__main__':
    download_data()
