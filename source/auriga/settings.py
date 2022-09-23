

class Settings:
    """
    A class used to manage the general configurations of this project.

    Attributes
    ----------
    first_snapshot : int
        The snapshot in which to start the analysis.
    repo_name : str
        The name of the repository of this project.
    """

    def __init__(self) -> None:
        self.first_snapshot = 15
        self.repo_name = 'auriga-metals'
