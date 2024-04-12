import json
import os
import shutil
from pathlib import Path

from epochalyst.logging.section_separator import print_section_separator

DEPENDENCIES_SAVE_PATH = Path('dependencies')
SOURCE_CODE_SAVE_PATH = Path('source-code')
SOURCE_CODE_PATH = Path('../')

# You can specify tm hashes here to exclude them from the source code dataset.
TM_HASH = [
    # "...",
]


def verify_config():
    print_section_separator("Verify the config files.")
    # Check if kaggle API is setup and installed
    try:
        import kaggle
    except OSError as e:
        new_message = f"To install the Kaggle API key, go to your kaggle profile -> Settings -> Create New API Token."
        raise OSError(new_message) from e

    # Check if the config file is present and filled in
    configs = Path("config")
    source_path = configs / "source.json"
    dependencies_path = configs / "dependencies.json"

    # load the config files to json
    source_config = json.load(open(source_path))
    dependencies_config = json.load(open(dependencies_path))

    # Check if dependencies is setup
    if source_config["title"] == ".placeholder":
        print("You have not setup the title of the source.json file. Let's do that now.")
        title = input("  - Enter the title of the dataset: ")
        source_config["title"] = title
    if source_config["id"] == ".placeholder/.placeholder":
        print(
            "You have not setup the id of the source.json file. Let's do that now. The username and id can be found in the URL of the dataset.\nFor example, in the URL https://www.kaggle.com/username/dataset-id, the username is 'username' and the id is 'dataset-id'")
        username = input("  - Enter the username of the owner of the source dataset: ")
        id = input("  - Enter the id of the source dataset: ")
        source_config["id"] = f"{username}/{id}"

    # Check if dependencies is setup
    if dependencies_config["title"] == ".placeholder":
        print("You have not setup the title of the dependencies.json file. Let's do that now.")
        title = input("  - Enter the title of the dataset: ")
        dependencies_config["title"] = title
    if dependencies_config["id"] == ".placeholder/.placeholder":
        print(
            "You have not setup the id of the dependencies.json file. Let's do that now. The username and id can be found in the URL of the dataset.\nFor example, in the URL https://www.kaggle.com/username/dataset-id, the username is 'username' and the id is 'dataset-id'")
        username = input("  - Enter the username of the owner of the dependencies dataset: ")
        id = input("  - Enter the id of the dependencies dataset ")
        dependencies_config["id"] = f"{username}/{id}"

    # Write back the updated source_config and dataset_config
    with open(source_path, "w") as f:
        json.dump(source_config, f)
    with open(dependencies_path, "w") as f:
        json.dump(dependencies_config, f)

    print("Config files have been verified.")


def update_dependencies():
    print_section_separator("Update the dependencies.")

    if os.path.exists(DEPENDENCIES_SAVE_PATH):
        print('Cleaning the dependencies folder')
        for filename in os.listdir(DEPENDENCIES_SAVE_PATH):
            file_path = os.path.join(DEPENDENCIES_SAVE_PATH, filename)
            if filename != 'tmp':
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
    else:
        os.makedirs(DEPENDENCIES_SAVE_PATH)

    print('Copying the requirements.txt file and excluding -e')
    with open(SOURCE_CODE_PATH / 'requirements.txt', 'r') as f:
        lines = f.readlines()
    with open(DEPENDENCIES_SAVE_PATH / 'requirements.txt', 'w') as f:
        for line in lines:
            if line.startswith('-e'):
                continue
            if line.startswith('kaggle'):
                continue
            f.write(line)

    if not os.path.exists(DEPENDENCIES_SAVE_PATH / 'tmp'):
        os.makedirs(DEPENDENCIES_SAVE_PATH / 'tmp')
    print('Downloading the dependencies')
    if not os.path.exists(DEPENDENCIES_SAVE_PATH / 'tmp'):
        os.makedirs(DEPENDENCIES_SAVE_PATH / 'tmp')
    # Run pip command
    os.system(f'pip download -r {DEPENDENCIES_SAVE_PATH / "requirements.txt"} -d {DEPENDENCIES_SAVE_PATH / "tmp"}')

    print('Zipping the downloaded dependencies')
    shutil.make_archive(DEPENDENCIES_SAVE_PATH / 'dependencies', 'zip', DEPENDENCIES_SAVE_PATH / 'tmp')
    shutil.move(DEPENDENCIES_SAVE_PATH / 'dependencies.zip', DEPENDENCIES_SAVE_PATH / 'dependencies.no_unzip')
    shutil.rmtree(DEPENDENCIES_SAVE_PATH / 'tmp')

    print('Copying the dataset-metadata.json file')
    shutil.copy('config/dependencies.json', DEPENDENCIES_SAVE_PATH / 'dataset-metadata.json')

    print('Excluding --find-files in requirements.txt')
    with open(DEPENDENCIES_SAVE_PATH / 'requirements.txt', 'r') as f:
        lines = f.readlines()
    with open(DEPENDENCIES_SAVE_PATH / 'requirements.txt', 'w') as f:
        for line in lines:
            if line.startswith('--find-links'):
                continue
            f.write(line)

    print('Done')

    # Upload the dataset
    os.system(f'kaggle datasets version -p ./dependencies -m "Update Dependencies"')


def update_source():
    if os.path.exists(SOURCE_CODE_SAVE_PATH):
        shutil.rmtree(SOURCE_CODE_SAVE_PATH)
    os.mkdir(SOURCE_CODE_SAVE_PATH)

    # Copy Source Code to submission/source_code
    relevant_files = ['src/', 'conf/', 'submit.py']
    if len(TM_HASH) == 0:
        relevant_files.append('tm/')
    else:
        for hash in TM_HASH:
            found_one = False
            tm = os.listdir(SOURCE_CODE_PATH / 'tm')
            for file in tm:
                if file.startswith(hash):
                    found_one = True
                    relevant_files.append('tm/' + file)
            if not found_one:
                print(f'No files found with hash: {hash}')
                exit(1)

    # Exclude __pycache__ from copying
    exluded_files = ['__pycache__']

    # Copy relevant files to tmp
    for file in relevant_files:
        if os.path.isdir(SOURCE_CODE_PATH / file):
            # Copy directory, skip excluded files with shutil
            shutil.copytree(SOURCE_CODE_PATH / file, SOURCE_CODE_SAVE_PATH / "tmp" / file, ignore=shutil.ignore_patterns(*exluded_files))
        else:
            # Copy file and create directories if not exist
            os.makedirs(SOURCE_CODE_SAVE_PATH / "tmp" / os.path.dirname(file), exist_ok=True)
            shutil.copy(SOURCE_CODE_PATH / file, SOURCE_CODE_SAVE_PATH / "tmp" / file)

    # Zip source_code
    shutil.make_archive(SOURCE_CODE_SAVE_PATH / 'source-code', 'zip', SOURCE_CODE_SAVE_PATH / "tmp")
    shutil.rmtree(SOURCE_CODE_SAVE_PATH / "tmp")

    # # Copy dataset-metadata.json to submission
    shutil.copy('dataset-metadata-source-code.json', SOURCE_CODE_SAVE_PATH / 'dataset-metadata.json')

    print('Submission files saved to source_code')

    os.system(f'kaggle datasets version -p ./source-code -m "Update Source Code"')


def manage_datasets():
    # Verify the config
    verify_config()

    # Update dependencies
    update_dep = input("Would you like to update the dependencies? (y/n): ").lower()

    if update_dep == "y":
        update_dependencies()

    # Update the dataset
    update_s = input("Would you like to update the source code? (y/n): ").lower()
    if update_s == "y":
        update_source()


if __name__ == "__main__":
    manage_datasets()
