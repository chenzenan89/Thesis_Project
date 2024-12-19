import json


class SettingsManager:

    def __init__(
            self,
            settings_file="/home/chen/Thesis_Project/app/config/setting.json"):
        self.settings_file = settings_file
        self.settings = self._load_settings()

    def _load_settings(self):
        try:
            with open(self.settings_file, "r") as file:
                return json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"setting file not found: {self.settings_file}")
        except json.JSONDecodeError as e:
            raise ValueError(f"setting file parsing failed: {str(e)}")

    def get_project_config(self, project_name):
        projects = self.settings.get(project_name)
        if not projects:
            raise ValueError(f"can not find '{project_name}'")
        return projects
