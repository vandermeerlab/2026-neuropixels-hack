import os
import ftplib
import paramiko
import re
from datetime import datetime
import getpass
from pathlib import Path
import json
import sys
import argparse

def load_config(config_file):
    """Load configuration from a JSON file"""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        print(f"Loaded configuration from {config_file}")
        return config
    except FileNotFoundError:
        print(f"Config file {config_file} not found. Using default configuration.")
        return {
            "host": "localhost",
            "user": "username",
            "password": "",  # Empty to prompt securely
            "port": 22,
            "protocol": "scp",
            "remote_base_dir": "",
            "remote_intermediate_path": None,
            "local_base_dir": "",
            "file_suffixes": [".csv"]  # Changed from file_extensions to file_suffixes
        }
    except json.JSONDecodeError:
        print(f"Error parsing config file {config_file}. Using default configuration.")
        return {
            "host": "localhost",
            "user": "username",
            "password": "",
            "port": 22,
            "protocol": "scp",
            "remote_base_dir": "",
            "remote_intermediate_path": None,
            "local_base_dir": "",
            "file_suffixes": [".csv"]  # Changed from file_extensions to file_suffixes
        }

def matches_suffix(filename, file_suffixes):
    """Check if filename matches any of the specified suffixes"""
    if not file_suffixes:  # If no suffixes specified, match all files
        return True
    
    filename_lower = filename.lower()
    for suffix in file_suffixes:
        suffix_lower = suffix.lower()
        if filename_lower.endswith(suffix_lower):
            return True
    return False

def get_file_info_sftp(sftp, filepath):
    """Get file creation and modification time using SFTP"""
    stat = sftp.stat(filepath)
    return {
        'size': stat.st_size,
        'mtime': datetime.fromtimestamp(stat.st_mtime),
        'atime': datetime.fromtimestamp(stat.st_atime)
    }

def get_file_info_ftp(ftp, filepath):
    """Get file modification time using FTP"""
    # FTP doesn't provide creation time, only modification time
    try:
        mod_time_str = ftp.sendcmd(f'MDTM {filepath}')
        # Parse the MDTM response - format typically like '213 YYYYMMDDhhmmss'
        if mod_time_str.startswith('213 '):
            time_str = mod_time_str[4:]
            try:
                mod_time = datetime.strptime(time_str, '%Y%m%d%H%M%S')
                return {
                    'mtime': mod_time,
                    'size': ftp.size(filepath)
                }
            except ValueError:
                pass
    except:
        pass
    
    # Fallback if MDTM failed
    return {
        'mtime': datetime.now(),
        'size': ftp.size(filepath)
    }

def prompt_overwrite(local_path, remote_path, local_info, remote_info):
    """Prompt user whether to overwrite existing file"""
    print("\nFile already exists:")
    print(f"Path: {local_path}")
    
    print("\nLocal file:")
    print(f"  Size: {local_info.get('size', 'Unknown')} bytes")
    print(f"  Modified: {local_info.get('mtime', 'Unknown')}")
    
    print("\nRemote file:")
    print(f"  Size: {remote_info.get('size', 'Unknown')} bytes")
    print(f"  Modified: {remote_info.get('mtime', 'Unknown')}")
    
    while True:
        response = input("\nOverwrite? (y/n): ").lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("Please answer 'y' or 'n'")

def download_via_ftp(config):
    """Download files using FTP"""
    # Extract config parameters
    host = config.get("host")
    port = config.get("port", 21)
    user = config.get("user")
    password = config.get("password", "")
    remote_base_dir = config.get("remote_base_dir")
    remote_intermediate_path = config.get("remote_intermediate_path")
    local_base_dir = config.get("local_base_dir")
    file_suffixes = config.get("file_suffixes", config.get("file_extensions", [".csv"]))  # Backwards compatibility
    
    # Connect to FTP server
    print(f"Connecting to FTP server {host}:{port}...")
    ftp = ftplib.FTP()
    ftp.connect(host, port)
    
    # If password is empty, prompt for it securely
    pwd = password if password else getpass.getpass('Enter FTP password: ')
    ftp.login(user, pwd)
    print("Login successful.")
    
    # Change to the remote base directory
    try:
        ftp.cwd(remote_base_dir)
        print(f"Changed to remote directory: {remote_base_dir}")
    except ftplib.error_perm as e:
        print(f"Error accessing remote directory: {e}")
        ftp.quit()
        return
    
    # Get list of MABC folders
    mabc_folders = []
    ftp.dir(lambda line: mabc_folders.append(line.split()[-1]))
    
    # Filter to only include folders that match MABC pattern
    mabc_pattern = re.compile(r'^M\d+$')
    mabc_folders = [folder for folder in mabc_folders if mabc_pattern.match(folder)]
    
    if not mabc_folders:
        print("No MABC folders found.")
        ftp.quit()
        return
    
    print(f"Found {len(mabc_folders)} MABC folders.")
    
    # Process each MABC folder
    for mabc_folder in mabc_folders:
        print(f"\nProcessing {mabc_folder}...")
        
        # Check if local MABC folder exists
        local_mabc_path = Path(local_base_dir) / mabc_folder
        if not local_mabc_path.exists():
            print(f"Local directory {local_mabc_path} doesn't exist, skipping...")
            continue
        
        # Enter the MABC folder
        try:
            ftp.cwd(mabc_folder)
        except ftplib.error_perm as e:
            print(f"Error accessing folder {mabc_folder}: {e}")
            continue
            
        # If remote_intermediate_path is specified, navigate to it
        if remote_intermediate_path:
            try:
                ftp.cwd(remote_intermediate_path)
                print(f"Changed to intermediate directory: {remote_intermediate_path}")
            except ftplib.error_perm as e:
                print(f"Error accessing intermediate directory {remote_intermediate_path}: {e}")
                ftp.cwd('..')  # Go back to base directory
                continue
        
        # Get list of date-specific folders
        date_folders = []
        ftp.dir(lambda line: date_folders.append(line.split()[-1]))
        
        # Filter to match the MABC-YYYY-MM-DD pattern
        date_pattern = re.compile(r'^' + mabc_folder + r'-\d{4}-\d{2}-\d{2}$')
        date_folders = [folder for folder in date_folders if date_pattern.match(folder)]
        
        if not date_folders:
            print(f"No date folders found in {mabc_folder}" + 
                  (f"/{remote_intermediate_path}" if remote_intermediate_path else "") + ".")
            # Go back to base directory
            if remote_intermediate_path:
                ftp.cwd('..')
            ftp.cwd('..')
            continue
        
        # Process each date folder
        for date_folder in date_folders:
            print(f"  Processing {date_folder}...")
            
            # Check if local date folder exists
            local_date_path = local_mabc_path / date_folder
            if not local_date_path.exists():
                print(f"  Local directory {local_date_path} doesn't exist, skipping...")
                continue
            
            # Enter the date folder
            try:
                ftp.cwd(date_folder)
            except ftplib.error_perm as e:
                print(f"  Error accessing folder {date_folder}: {e}")
                continue
            
            # Get files in the date folder
            files = []
            ftp.dir(lambda line: files.append(line.split()[-1]))
            
            # Download files with matching suffixes
            for filename in files:
                if matches_suffix(filename, file_suffixes):
                    remote_file_path = filename  # Current directory in FTP
                    local_file_path = local_date_path / filename
                    
                    # Check if file already exists locally
                    if local_file_path.exists():
                        # Get file info
                        local_info = {
                            'size': local_file_path.stat().st_size,
                            'mtime': datetime.fromtimestamp(local_file_path.stat().st_mtime)
                        }
                        remote_info = get_file_info_ftp(ftp, remote_file_path)
                        
                        # Prompt user for overwrite
                        if not prompt_overwrite(local_file_path, remote_file_path, local_info, remote_info):
                            print(f"    Skipping {filename}...")
                            continue
                    
                    print(f"    Downloading {filename}...")
                    try:
                        with open(local_file_path, 'wb') as file:
                            ftp.retrbinary(f"RETR {filename}", file.write)
                        print(f"    Downloaded {filename} successfully.")
                    except Exception as e:
                        print(f"    Error downloading {filename}: {e}")
            
            # Go back to MABC or intermediate folder
            ftp.cwd('..')
        
        # Go back to base directory
        if remote_intermediate_path:
            ftp.cwd('..')
        ftp.cwd('..')
    
    # Close FTP connection
    ftp.quit()
    print("\nDownload process completed via FTP.")

def download_via_scp(config):
    """Download files using SCP/SFTP but with direct Windows commands for directory listing"""
    # Extract config parameters
    host = config.get("host")
    port = config.get("port", 22)
    user = config.get("user")
    password = config.get("password", "")
    remote_base_dir = config.get("remote_base_dir")
    remote_intermediate_path = config.get("remote_intermediate_path")
    local_base_dir = config.get("local_base_dir")
    file_suffixes = config.get("file_suffixes", config.get("file_extensions", [".csv"]))  # Backwards compatibility
    
    print(f"Connecting to SSH server {host}:{port}...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    
    try:
        # Connect to server
        pwd = password if password else getpass.getpass('Enter SSH password: ')
        ssh.connect(hostname=host, port=port, username=user, password=pwd)
        print("SSH connection established.")
        
        # Define the remote path 
        remote_path = remote_base_dir.replace('\\', '/') 
        
        # Detect if we're on Windows remote system
        stdin, stdout, stderr = ssh.exec_command('uname -s')
        uname_output = stdout.read().decode().strip()
        is_windows = not uname_output or 'win' in uname_output.lower()
        
        if is_windows:
            print("Detected Windows remote system")
            path_sep = '\\'
        else:
            print("Detected Unix/Linux remote system")
            path_sep = '/'
        
        # Use appropriate command for directory listing
        if is_windows:
            list_cmd = f'dir "{remote_base_dir}"'
        else:
            list_cmd = f'ls -la "{remote_base_dir}"'
        
        # List MABC folders
        print(f"Listing directory: {remote_base_dir}")
        stdin, stdout, stderr = ssh.exec_command(list_cmd)
        dir_output = stdout.read().decode()
        
        # Parse the dir output to extract folder names
        mabc_folders = []
        if is_windows:
            for line in dir_output.splitlines():
                if "d-----" in line:  # Windows directory indicator
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        folder_name = parts[-1]  # Folder name is the last part
                        if re.match(r'^M\d+$', folder_name):
                            mabc_folders.append(folder_name)
        else:
            for line in dir_output.splitlines():
                if line.startswith('d'):  # Unix directory indicator
                    parts = line.strip().split()
                    if len(parts) >= 9:
                        folder_name = parts[8]  # Folder name position in ls -la output
                        if re.match(r'^M\d+$', folder_name):
                            mabc_folders.append(folder_name)
        
        if not mabc_folders:
            print("No MABC folders found after parsing directory output.")
            ssh.close()
            return
        
        print(f"Found {len(mabc_folders)} MABC folders: {mabc_folders}")
        
        # Create SFTP client for file transfers only
        sftp = ssh.open_sftp()
        
        # Process each MABC folder
        for mabc_folder in mabc_folders:
            print(f"\nProcessing {mabc_folder}...")
            
            # Check if local MABC folder exists
            local_mabc_path = Path(local_base_dir) / mabc_folder
            if not local_mabc_path.exists():
                print(f"Local directory {local_mabc_path} doesn't exist, skipping...")
                continue
            
            # Determine path to date folders
            if remote_intermediate_path:
                # If intermediate path is specified, look for date folders there
                date_folder_path = f"{remote_base_dir}{path_sep}{mabc_folder}{path_sep}{remote_intermediate_path}"
            else:
                # Otherwise, look directly in the MABC folder
                date_folder_path = f"{remote_base_dir}{path_sep}{mabc_folder}"
            
            # List date folders
            if is_windows:
                list_cmd = f'dir "{date_folder_path}"'
            else:
                list_cmd = f'ls -la "{date_folder_path}"'
            
            stdin, stdout, stderr = ssh.exec_command(list_cmd)
            date_folder_output = stdout.read().decode()
            error_output = stderr.read().decode()
            
            if error_output:
                print(f"Error listing {date_folder_path}: {error_output}")
                continue
            
            # Parse output to find date folders
            date_folders = []
            if is_windows:
                for line in date_folder_output.splitlines():
                    if "d-----" in line:  # Directory indicator
                        parts = line.strip().split()
                        if len(parts) >= 4:
                            folder_name = parts[-1]
                            if re.match(r'^' + mabc_folder + r'-\d{4}-\d{2}-\d{2}$', folder_name):
                                date_folders.append(folder_name)
            else:
                for line in date_folder_output.splitlines():
                    if line.startswith('d'):  # Unix directory indicator
                        parts = line.strip().split()
                        if len(parts) >= 9:
                            folder_name = parts[8]
                            if re.match(r'^' + mabc_folder + r'-\d{4}-\d{2}-\d{2}$', folder_name):
                                date_folders.append(folder_name)
            
            if not date_folders:
                print(f"No date folders found in {date_folder_path}.")
                continue
            
            print(f"Found date folders: {date_folders}")
            
            # Process each date folder
            for date_folder in date_folders:
                print(f"  Processing {date_folder}...")
                
                # Check if local date folder exists
                local_date_path = local_mabc_path / date_folder
                if not local_date_path.exists():
                    print(f"  Local directory {local_date_path} doesn't exist, skipping...")
                    continue
                
                # Use SSH command to list files
                file_folder_path = f"{date_folder_path}{path_sep}{date_folder}"
                if is_windows:
                    list_cmd = f'dir "{file_folder_path}"'
                else:
                    list_cmd = f'ls -la "{file_folder_path}"'
                
                stdin, stdout, stderr = ssh.exec_command(list_cmd)
                files_output = stdout.read().decode()
                error_output = stderr.read().decode()
                
                if error_output:
                    print(f"  Error listing {file_folder_path}: {error_output}")
                    continue
                
                # Parse output to find files
                files = []
                if is_windows:
                    for line in files_output.splitlines():
                        if "-a----" in line:  # File indicator
                            parts = line.strip().split()
                            if len(parts) >= 4:
                                filename = parts[-1]  # Filename is the last item
                                files.append(filename)
                else:
                    for line in files_output.splitlines():
                        if not line.startswith('d'):  # Not a directory
                            parts = line.strip().split()
                            if len(parts) >= 9:
                                filename = parts[8]
                                files.append(filename)
                
                # Download files with matching suffixes
                for filename in files:
                    if matches_suffix(filename, file_suffixes):
                        # Construct SFTP paths with correct separators
                        if remote_intermediate_path:
                            remote_file_path = f"{remote_path}/{mabc_folder}/{remote_intermediate_path}/{date_folder}/{filename}".replace('\\', '/')
                        else:
                            remote_file_path = f"{remote_path}/{mabc_folder}/{date_folder}/{filename}".replace('\\', '/')
                        
                        local_file_path = local_date_path / filename
                        
                        # Check if file already exists locally
                        if local_file_path.exists():
                            local_info = {
                                'size': local_file_path.stat().st_size,
                                'mtime': datetime.fromtimestamp(local_file_path.stat().st_mtime)
                            }
                            
                            # Get remote file info
                            if is_windows:
                                file_info_cmd = f'dir "{file_folder_path}{path_sep}{filename}"'
                            else:
                                file_info_cmd = f'ls -la "{file_folder_path}/{filename}"'
                            
                            stdin, stdout, stderr = ssh.exec_command(file_info_cmd)
                            file_info = stdout.read().decode()
                            
                            # Parse file info
                            remote_info = {'size': 'Unknown', 'mtime': 'Unknown'}
                            if is_windows:
                                for line in file_info.splitlines():
                                    if filename in line and "-a----" in line:
                                        parts = line.strip().split()
                                        if len(parts) >= 4:
                                            try:
                                                size = int(parts[-2])
                                                remote_info = {
                                                    'size': size,
                                                    'mtime': ' '.join(parts[1:3])  # Date and time
                                                }
                                            except ValueError:
                                                pass
                            else:
                                for line in file_info.splitlines():
                                    if not line.startswith('d'):  # Not a directory
                                        parts = line.strip().split()
                                        if len(parts) >= 9:
                                            try:
                                                size = int(parts[4])
                                                date_str = ' '.join(parts[5:8])
                                                remote_info = {
                                                    'size': size,
                                                    'mtime': date_str
                                                }
                                            except ValueError:
                                                pass
                            
                            # Prompt user for overwrite
                            if not prompt_overwrite(local_file_path, remote_file_path, local_info, remote_info):
                                print(f"    Skipping {filename}...")
                                continue
                        
                        print(f"    Downloading {filename}...")
                        try:
                            sftp.get(remote_file_path, str(local_file_path))
                            print(f"    Downloaded {filename} successfully.")
                        except Exception as e:
                            print(f"    Error downloading {filename}: {e}")
                            print(f"    Remote path: {remote_file_path}")
                            print(f"    Local path: {local_file_path}")
        
        # Close connections
        sftp.close()
        ssh.close()
        print("\nDownload process completed via SSH/SCP.")
        
    except Exception as e:
        print(f"SSH connection error: {e}")
        import traceback
        traceback.print_exc()

def main():
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Download files from remote server using FTP or SCP')
    parser.add_argument('config_path', help='Path to the transfer_config.json file')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if config file exists
    config_path = Path(args.config_path)
    if not config_path.exists():
        print(f"Error: Configuration file '{args.config_path}' not found.")
        sys.exit(1)
    
    # Load configuration
    config = load_config(config_path)
    
    # Create base local directory if it doesn't exist
    local_base = Path(config.get("local_base_dir"))
    if not local_base.exists():
        local_base.mkdir(parents=True)
        print(f"Created local base directory: {local_base}")
    
    protocol = config.get("protocol", "scp").lower()
    if protocol == "ftp":
        download_via_ftp(config)
    elif protocol == "scp":
        download_via_scp(config)
    else:
        print(f"Unsupported protocol: {protocol}. Please use 'ftp' or 'scp'.")

if __name__ == "__main__":
    main()