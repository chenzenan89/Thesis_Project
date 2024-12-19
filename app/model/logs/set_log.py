import os
import logging

# Path where log files will be saved
LOG_PATH = '/home/chen/Thesis_Project/app/model/logs/log_data'


class logger:
    """
    Handles logging for a specific project.
    """

    def __init__(self, project_name: str):
        """
        Set up the logger for the project.
        """
        logger = logging.getLogger(project_name)
        if not logger.hasHandlers():
            # Create a log file and attach a file handler
            log_file = os.path.join(LOG_PATH, f"{project_name}.log")
            handler = logging.FileHandler(log_file, mode="a")
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

        # Save the log file path for other stuff
        self.path = os.path.join(LOG_PATH, f"{project_name}.log")
        self.log = logger

    def update_first_line(self, new_content):
        """
        Change the first line of the log file to whatever you want.
        """
        if os.path.exists(self.path):
            with open(self.path, "r+") as f:
                lines = f.readlines()
                # Update the first line or add it if the file is empty
                if lines:
                    lines[0] = new_content + "\n"
                else:
                    lines.append(new_content + "\n")
                # Write the updated content back
                f.seek(0)
                f.writelines(lines)
                f.truncate()
