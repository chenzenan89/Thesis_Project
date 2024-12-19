from tinydb import TinyDB, Query


class Database:
    """
    A simple wrapper around TinyDB for basic database operations.
    """

    def __init__(self, path: str):
        """
        Initialize the database at the given path.
        """
        self.path = path
        self.db = TinyDB(self.path)  # Create or connect to the database file
        self.Name = Query()  # Query object for filtering data

    def insert(self, user: dict):
        """
        Add a new record to the database.
        Args:
            user (dict): Data to insert.
        Returns:
            int: ID of the inserted record.
        """
        return self.db.insert(user)

    def all_check(self):
        """
        Get all records in the database.
        Returns:
            list: A list of all records.
        """
        return self.db.all()

    def search(self, name):
        """
        Search for records by name.
        Args:
            name (str): The name to search for.
        Returns:
            list: Matching records.
        """
        return self.db.search(self.Name.name == name)
