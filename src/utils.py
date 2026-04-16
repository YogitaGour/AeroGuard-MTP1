import yaml
from pathlib import Path

def load_config(config_path="config/settings.yaml"):
    config_path = Path(config_path)
    print(f"Looking for config at: {config_path.absolute()}")

    if not config_path.exists():
        print(f"ERROR: Config file not found at {config_path.absolute()}")
        return None

    try:
        with open(config_path, 'r') as f:
            content = f.read()
            print(f"Config file content length: {len(content)} chars")
            config = yaml.safe_load(content)
            if config is None:
                print("ERROR: YAML parsing returned None (empty file?)")
                return None
            print("Config loaded successfully!")
            return config
    except Exception as e:
        print(f"ERROR loading config: {e}")
        return None