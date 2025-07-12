import requests
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv
import os
import pythoncom
from win32com.shell import shell, shellcon
import gc
import time

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
    def __init__(self, api_url, api_key, verbose=False, max_retries=3, retry_delay=3):
        self.api_url = api_url
        self.api_key = api_key
        self.headers = {
            'Accept': 'application/json',
            'x-api-key': self.api_key,
        }
        self.verbose = verbose
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def get_albums(self):
        r = requests.get(f'{self.api_url}/albums', headers=self.headers)
        if r.status_code == 200:
            return r.json()
        if self.verbose:
            print(f"Failed to get albums: {r.status_code} {r.text}")
        return []

    def find_album_id(self, target_name):
        albums = self.get_albums()
        for album in albums:
            if album.get('albumName') == target_name:
                return album.get('id')
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

        attempts = 0
        while attempts < self.max_retries:
            attempts += 1
            try:
                with open(file_path, 'rb') as f:
                    # File handle is passed to requests; file streamed, not fully read into memory
                    files = {'assetData': (file_path.name, f, 'application/octet-stream')}
                    r = requests.post(f'{self.api_url}/assets', headers=self.headers, data=data, files=files, timeout=60)
                if r.status_code in (200, 201):
                    return r.json().get('id')
                else:
                    if self.verbose:
                        print(f"Upload fail {file_path.name} [{r.status_code}]: {r.text}")
            except MemoryError as me:
                print(f"MemoryError encountered uploading {file_path.name}, attempt {attempts} of {self.max_retries}")
                gc.collect()
                time.sleep(self.retry_delay)
            except requests.exceptions.RequestException as e:
                if self.verbose:
                    print(f"Request error uploading {file_path.name}: {e}. Retrying ({attempts}/{self.max_retries})")
                time.sleep(self.retry_delay)
            except Exception as e:
                if self.verbose:
                    print(f"Error uploading {file_path.name}: {e}")
                break  # Unexpected error, no retry

        if self.verbose:
            print(f"Failed to upload {file_path.name} after {self.max_retries} attempts.")
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


def gather_files_from_folder(folder_path, immich: ImmichClient):
    """Collect all valid files in a folder (resolving shortcuts/symlinks)."""
    files = []
    for fp in folder_path.iterdir():
        try:
            # Ignore hidden/system files (optional)
            if fp.name.startswith('.'):
                continue

            # Regular file (.lnk handled separately)
            if fp.is_file() and not str(fp).lower().endswith('.lnk'):
                files.append(fp)
            # Windows shortcut .lnk file - resolve target
            elif str(fp).lower().endswith('.lnk'):
                target = resolve_shortcut(fp)
                if target:
                    target_path = Path(target)
                    if target_path.is_file():
                        files.append(target_path)
                    else:
                        if immich.verbose:
                            print(f"Resolved shortcut does not point to file: {fp}")
                else:
                    if immich.verbose:
                        print(f"Could not resolve shortcut: {fp}")
            # Symlink (not .lnk)
            elif fp.is_symlink():
                target = fp.resolve(strict=True)
                if target.is_file():
                    files.append(fp)
        except FileNotFoundError:
            if immich.verbose:
                print(f"Skipping broken symlink or missing file: {fp}")
            continue
        except Exception as e:
            if immich.verbose:
                print(f"Error accessing {fp}: {e}")
            continue
    return files


def handle_existing_album(immich: ImmichClient, album_name, default_choice=None):
    """Deal with naming/already-existing-album logic. Returns album ID or None on cancel."""

    album_id = immich.find_album_id(album_name)
    if not album_id:
        return None  # No conflict, album doesn't exist yet

    unique_name = f"{album_name}_2"

    if default_choice and default_choice.lower() in ['y', 'n', 's']:
        choice = default_choice.lower()
        print(f"Default choice '{choice}' selected for album '{album_name}'.")
    else:
        while True:
            choice = input(
                f"An album named '{album_name}' already exists (ID: {album_id}).\n"
                f"Would you like to add photos to this album (y), "
                f"or create a new album named '{unique_name}' (n), "
                "or cancel/skip (s): "
            ).strip().lower()
            if choice in ["y", "n", "s"]:
                break
            print("Please enter 'y', 'n', or 's'.")

    if choice == 'y':
        return album_id

    if choice == 'n':
        print(f"Creating new album: {unique_name}")
        new_album_id = immich.create_album(unique_name)
        if not new_album_id:
            print("Album creation failed, aborting.")
        return new_album_id

    print("Skipping.")
    return None  # Cancel


def import_single_folder_as_album(folder_path, immich: ImmichClient, default_choice=None):
    """Imports the given folder as one album. Handles prompting for name conflicts."""

    album_name = folder_path.name
    print(f"Preparing to import to album: {album_name}")

    files = gather_files_from_folder(folder_path, immich)
    if not files:
        print(f"No files or valid shortcuts/symlinks found in folder '{folder_path}', aborting.")
        return

    asset_ids = []
    for i, fp in enumerate(tqdm(files, desc=f"Uploading assets from {album_name}", unit="file")):
        aid = immich.upload_asset(fp)
        if aid:
            asset_ids.append(aid)

        # Periodic garbage collection every 10 uploads to free memory
        if i > 0 and i % 10 == 0:
            gc.collect()

    if not asset_ids:
        print(f"No assets uploaded from folder '{folder_path}', aborting.")
        return

    album_id = handle_existing_album(immich, album_name, default_choice)
    if album_id is None:
        # The album didn't exist or user cancelled
        if immich.find_album_id(album_name) is None:
            album_id = immich.create_album(album_name)
        # User might have cancelled or creation failed
        if album_id is None:
            print("Skipping (user cancelled or album creation failed).")
            return

    success = immich.add_assets_to_album(album_id, asset_ids)
    if success:
        print(f"Import completed for album '{album_name}'.")
    else:
        print(f"Failed to add assets to album '{album_name}'.")


def import_folder_as_album(folder_path, client, default_choice=None):
    """
    Import this folder (and optionally all subfolders) as albums.
    Only bottom-level folders containing files (and no subfolders) will be imported.
    """

    folder_path = Path(folder_path)
    if not folder_path.is_dir():
        print("Invalid directory:", folder_path)
        return

    print(f"Working on base folder recursively: {folder_path}")

    # Get all directories recursively
    all_dirs = [folder_path] + [p for p in folder_path.rglob('*') if p.is_dir()]

    # Filter to only bottom folders that contain files and no subdirectories
    all_folders = []
    for folder in all_dirs:
        try:
            # Using try/except in case permission error on some folders
            entries = list(folder.iterdir())
        except Exception as e:
            if client.verbose:
                print(f"Cannot access {folder}: {e}")
            continue

        has_files = any(f.is_file() for f in entries)
        has_subdirs = any(d.is_dir() for d in entries)
        if has_files and not has_subdirs:
            all_folders.append(folder)

    total = len(all_folders)
    for i, folder in enumerate(all_folders, start=1):
        print(f"{'*' * 20}>>> [{i}/{total}] Processing folder: {folder}")
        import_single_folder_as_album(folder, client, default_choice=default_choice)
        # Run garbage collection after each folder processed to release memory
        gc.collect()


if __name__ == "__main__":

    # Default user choice for album conflict: 'y' - add to existing album
    default_choice = "y"

    # Set your target folder path here
    folder_path = r"G:\photos_from_old_drive\2012"

    client = ImmichClient(IMMICH_API_URL, API_KEY, verbose=False)

    import_folder_as_album(folder_path, client, default_choice=default_choice)

    print("All folders processed.")
