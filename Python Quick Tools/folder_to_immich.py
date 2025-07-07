import requests
from pathlib import Path
from datetime import datetime
import difflib
from tqdm import tqdm


IMMICH_API_URL = r'http://192.168.1.38:2283/api'
API_KEY = 'HpmJhX7iQYFkvdsbibpdAv7KhbB6e0mzRDvHC92LPE'

class ImmichClient:
    def __init__(self, api_url, api_key):
        self.api_url = api_url
        self.api_key = api_key
        self.headers = {
            'Accept': 'application/json',
            'x-api-key': self.api_key,
        }
    
    def get_albums(self):
        r = requests.get(f'{self.api_url}/albums', headers=self.headers)
        return r.json() if r.status_code == 200 else []

    def find_album_id(self, target_name, cutoff=0.7):
        albums = self.get_albums()
        names = [a['albumName'] for a in albums]
        match = difflib.get_close_matches(target_name, names, n=1, cutoff=cutoff)
        if match:
            return next((a['id'] for a in albums if a['albumName'] == match[0]), None)
        return None

    def create_album(self, name, asset_ids=None, description=""):
        payload = {
            "albumName": name,
            "albumUsers": [],
            "assetIds": asset_ids or [],
            "description": description
        }
        hdrs = {**self.headers, 'Content-Type': 'application/json'}
        r = requests.post(f'{self.api_url}/albums', headers=hdrs, json=payload)
        if r.ok:
            return r.json().get('id')
        print(f"Failed to create album '{name}': {r.status_code} {r.text}")
        return None

    def upload_asset(self, file_path):
        stats = file_path.stat()
        data = {
            'deviceAssetId': f'{file_path.name}-{int(stats.st_mtime)}',
            'deviceId': 'python_script',
            'fileCreatedAt': datetime.fromtimestamp(stats.st_mtime).isoformat(),
            'fileModifiedAt': datetime.fromtimestamp(stats.st_mtime).isoformat(),
            'isFavorite': 'false',
        }
        with open(file_path, 'rb') as f:
            files = {'assetData': (file_path.name, f, 'application/octet-stream')}
            r = requests.post(f'{self.api_url}/assets', headers=self.headers, data=data, files=files)
        if r.status_code in (200, 201):
            return r.json().get('id')
        print(f"Failed to upload {file_path.name}: {r.status_code} {r.text}")
        return None

    def add_assets_to_album(self, album_id, asset_ids):
        url = f'{self.api_url}/albums/{album_id}/assets'
        payload = {"ids": asset_ids}
        hdrs = {**self.headers, 'Content-Type': 'application/json'}
        r = requests.put(url, headers=hdrs, json=payload)
        if r.status_code == 200:
            return True
        print(f"Failed to add assets to album {album_id}: {r.status_code} {r.text}")
        return False

def import_folder_as_album(folder_path, immich: ImmichClient):
    from tqdm import tqdm  # If you prefer, you can import this at the top of your file

    folder_path = Path(folder_path)
    if not folder_path.is_dir():
        print("Invalid directory:", folder_path)
        return

    album_name = folder_path.name
    print(f"Preparing to import to album: {album_name}")

    files = [fp for fp in folder_path.iterdir() if fp.is_file()]
    asset_ids = []
    for fp in tqdm(files, desc="Uploading assets", unit="file"):
        aid = immich.upload_asset(fp)
        if aid:
            asset_ids.append(aid)

    if not asset_ids:
        print("No assets uploaded, aborting.")
        return

    album_id = immich.find_album_id(album_name)
    if album_id:
        print(f"Found existing album ID: {album_id}, adding photos.")
    else:
        print(f"No album found. Creating new album: {album_name}")
        album_id = immich.create_album(album_name)
        if not album_id:
            print("Album creation failed, aborting.")
            return

    success = immich.add_assets_to_album(album_id, asset_ids)
    if success:
        print("Import completed.")
    else:
        print("Failed to add assets to album.")




if __name__ == "__main__":
    folder = r'G:\photos_from_old_drive\2006\Big Seattle Snow'
    client = ImmichClient(IMMICH_API_URL, API_KEY)
    import_folder_as_album(folder, client)
