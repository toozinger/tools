import requests
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv
import os
import pythoncom
from win32com.shell import shell, shellcon

load_dotenv()  # Load variables from .env file

IMMICH_API_URL = r'http://192.168.1.38:2283/api'
API_KEY = os.getenv('IMMICH_API_KEY')  # Load API key from environment

if not API_KEY:
    raise ValueError("API Key not found. Please set IMMICH_API_KEY in your .env file")

def resolve_shortcut(path):
    """
    Resolve a Windows shortcut (.lnk) file to its target path.

    Returns the resolved target path as a string, or None if cannot resolve.
    """
    if not os.path.exists(path):
        return None
    if not str(path).lower().endswith('.lnk'):
        return str(path)

    try:
        shortcut = pythoncom.CoCreateInstance(
            shell.CLSID_ShellLink, None,
            pythoncom.CLSCTX_INPROC_SERVER, shell.IID_IShellLink)
        persist_file = shortcut.QueryInterface(pythoncom.IID_IPersistFile)
        persist_file.Load(str(path))
        target_path, _ = shortcut.GetPath(shell.SLGP_RAWPATH)
        if target_path and os.path.exists(target_path):
            return target_path
    except Exception as e:
        print(f"Failed to resolve shortcut '{path}': {e}")

    return None


class ImmichClient:
    def __init__(self, api_url, api_key, verbose=False):
        self.api_url = api_url
        self.api_key = api_key
        self.headers = {
            'Accept': 'application/json',
            'x-api-key': self.api_key,
        }
        self.verbose = verbose
    
    def get_albums(self):
        r = requests.get(f'{self.api_url}/albums', headers=self.headers)
        return r.json() if r.status_code == 200 else []

    def find_album_id(self, target_name):
        albums = self.get_albums()
        for album in albums:
            if album['albumName'] == target_name:
                return album['id']
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
        if self.verbose:
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
    folder_path = Path(folder_path)
    if not folder_path.is_dir():
        print("Invalid directory:", folder_path)
        return

    album_name = folder_path.name
    print(f"Preparing to import to album: {album_name}")

    files = []
    for fp in folder_path.iterdir():
        try:
            # Case 1: Regular file or symlink to file (but not .lnk)
            if fp.is_file() and not str(fp).lower().endswith('.lnk'):
                files.append(fp)

            # Case 2: Windows shortcut .lnk file - resolve target
            elif str(fp).lower().endswith('.lnk'):
                target = resolve_shortcut(fp)
                if target:
                    target_path = Path(target)
                    if target_path.is_file():
                        files.append(target_path)
                    else:
                        if immich.verbose:
                            print(f"Resolved shortcut does not point to a file: {fp}")
                else:
                    if immich.verbose:
                        print(f"Could not resolve shortcut: {fp}")

            # Case 3: Symlink (non .lnk) - check resolved target is file
            elif fp.is_symlink():
                target = fp.resolve(strict=True)
                if target.is_file():
                    files.append(fp)

        except FileNotFoundError:
            # Broken symlink, ignore
            if immich.verbose:
                print(f"Skipping broken symlink or missing file: {fp}")
            continue

    if not files:
        print("No files or valid shortcuts/symlinks found in folder, aborting.")
        return

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
        unique_name = f"{album_name}_2"
        while True:
            choice = input(
                f"An album named '{album_name}' already exists "
                f"(ID: {album_id}).\n"
                "Would you like to add photos to this album (y), "
                f"or create a new album named '{unique_name} (n)?, or "
                "cancel/skip (s): "
            ).strip().lower()
            if choice in ["y", "n", "s"]:
                break
            print("Please enter 'y', 'n', or 's'.")

        if choice == "n":
            print(f"Creating new album: {unique_name}")
            album_id = immich.create_album(unique_name)
            if not album_id:
                print("Album creation failed, aborting.")
                return
        elif choice == "y":
            print(f"Adding to existing album: {album_id}")
        elif choice == "s":
            print("Skipping adding to an album")
            return
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
    # folders = [Path(r"H:\final_folder\3D Printing")]
    # If you want to scan directories inside a parent folder:
    super_path = Path(r"G:\photos_from_old_drive\2009")
    folders = [p for p in super_path.iterdir() if p.is_dir()]

    client = ImmichClient(IMMICH_API_URL, API_KEY, verbose=False)

    total = len(folders)
    for i, folder in enumerate(folders, start=1):
        print(f"{'*'*20}>>> [{i}/{total}] Processing folder: {folder.name}")
        import_folder_as_album(folder, client)
    
    print("All folders processed.")
