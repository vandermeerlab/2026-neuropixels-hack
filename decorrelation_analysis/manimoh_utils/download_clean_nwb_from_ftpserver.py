import os
import ftplib
import re
from datetime import datetime

# FTP server configuration
ftp_host = "129.170.31.30"           # FTP server IP address
ftp_user = "manishm"                 # FTP username
ftp_password = "$\"Z{]?"             # FTP password
remote_dir = "/datavault2/inProcess/odor-pixels/Task2_SWR/" # Remote top-level directory
local_dir = "/home/manishm/data/odor-pixels/"               # Local directory where files will be downloaded

# Main script
def main():
    # Connect to FTP server
    print(f"Connecting to FTP server {ftp_host}...")
    ftp = ftplib.FTP(ftp_host)
    ftp.login(ftp_user, ftp_password)
    print("Login successful.")
    
    # Change to the remote directory
    try:
        ftp.cwd(remote_dir)
        print(f"Changed to remote directory: {remote_dir}")
    except ftplib.error_perm as e:
        print(f"Error accessing remote directory: {e}")
        ftp.quit()
        return
    
    # Get list of MABC folders
    mabc_folders = []
    ftp.dir(lambda line: mabc_folders.append(line.split()[-1]))
    
    # Filter to only include folders that match MABC pattern (where A, B, C are whole numbers)
    mabc_pattern = re.compile(r'^M\d+\d+\d+$')
    mabc_folders = [folder for folder in mabc_folders if mabc_pattern.match(folder)]
    
    if not mabc_folders:
        print("No MABC folders found.")
        ftp.quit()
        return
    
    print(f"Found {len(mabc_folders)} MABC folders.")
    
    # Process each MABC folder
    for mabc_folder in mabc_folders:
        print(f"\nProcessing {mabc_folder}...")
        
        # Enter the MABC folder
        try:
            ftp.cwd(mabc_folder)
        except ftplib.error_perm as e:
            print(f"Error accessing folder {mabc_folder}: {e}")
            continue
        
        # Enter the preprocessed folder
        try:
            ftp.cwd("preprocessed")
        except ftplib.error_perm as e:
            print(f"Error accessing 'preprocessed' folder in {mabc_folder}: {e}")
            ftp.cwd('..')  # Go back to MABC folder
            continue
            
        # Get list of date-specific folders
        date_folders = []
        ftp.dir(lambda line: date_folders.append(line.split()[-1]))
        
        # Filter to match the MABC-YYYY-MM-DD pattern
        date_pattern = re.compile(r'^' + mabc_folder + r'-\d{4}-\d{2}-\d{2}$')
        date_folders = [folder for folder in date_folders if date_pattern.match(folder)]
        
        if not date_folders:
            print(f"No date folders found in {mabc_folder}/preprocessed.")
            ftp.cwd('../..')  # Go back to top level
            continue
        
        # Process each date folder
        for date_folder in date_folders:
            print(f"  Processing {date_folder}...")
            
            # Enter the date folder
            try:
                ftp.cwd(date_folder)
            except ftplib.error_perm as e:
                print(f"  Error accessing folder {date_folder}: {e}")
                continue
            
            # Create local directory structure
            local_folder_path = os.path.join(local_dir, mabc_folder, date_folder)
            os.makedirs(local_folder_path, exist_ok=True)
            
            # Get files in the date folder
            files = []
            ftp.dir(lambda line: files.append(line.split()[-1]))
            
            # Define patterns for the two file types we want
            key_file_pattern = re.compile(r'^' + mabc_folder + r'_\d{4}_\d{2}_\d{2}_keys\.m$')
            test_file_pattern = re.compile(r'^test_\d{4}-\d{2}-\d{2}\.nwb$')
            
            # Download files that match our patterns
            for filename in files:
                if key_file_pattern.match(filename) or test_file_pattern.match(filename):
                    local_file_path = os.path.join(local_folder_path, filename)
                    print(f"    Downloading {filename}...")
                    
                    try:
                        with open(local_file_path, 'wb') as file:
                            ftp.retrbinary(f"RETR {filename}", file.write)
                        print(f"    Downloaded {filename} successfully.")
                    except Exception as e:
                        print(f"    Error downloading {filename}: {e}")
            
            # Go back to preprocessed folder
            ftp.cwd('..')
        
        # Go back to top level directory
        ftp.cwd('../..')
    
    # Close FTP connection
    ftp.quit()
    print("\nDownload process completed.")

if __name__ == "__main__":
    main()
