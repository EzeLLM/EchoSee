import yaml
import inspect

def open_yaml(file_path, key=None):
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
            if key is not None:
                return data.get(key)
            else:
                return data
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing {file_path}: {e}")
        return None
    except AttributeError:
        print(f"Key {key} not found in file {file_path}.")
        return None

