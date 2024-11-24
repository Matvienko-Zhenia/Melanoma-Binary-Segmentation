import gdown
from pathlib import Path


class WeightsProvider:
    """  
    A class to download and provide weights for models.  
  
    :ivar file_id: The Google Drive file ID for the weights.  
    :vartype file_id: str  
    :ivar weights_path: The local path where weights will be stored.  
    :vartype weights_path: Path  
    """  
    def __init__(
            self,
            file_id: str,
            weights_path: Path
    ) -> None:
        """  
        Constructs all the necessary attributes for the WeightsProvider object.  
  
        :param file_id: The Google Drive file ID for the weights.  
        :type file_id: str  
        :param weights_path: The local path where weights will be stored.  
        :type weights_path: Path  
        """ 
        self._file_id = file_id
        self._url = f"https://drive.google.com/uc?id={self._file_id}"
        self._weights_path = weights_path

    def get_weights_path(self) -> Path:
        """  
        Returns the path where weights are stored.  
  
        :returns: The local path where weights are stored.  
        :rtype: Path  
        """  
        return self._weights_path

    def download_file(self, file_name: str, force: bool = False) -> str:
        """  
        Downloads the file from Google Drive.  
  
        :param file_name: The name of the file to download.  
        :type file_name: str  
        :param force: If True, forces the download even if the file already exists (default is False).  
        :type force: bool, optional  
        :returns: The path to the downloaded file.  
        :rtype: str  
        """  
        if (not (self._weights_path / file_name).is_file()) | force:
            (self._weights_path / file_name.rsplit("/", maxsplit=1)[0]).mkdir(parents=True, exist_ok=True)
            output = str(self._weights_path / file_name)
            return gdown.download(self._url, output, quiet=False)
