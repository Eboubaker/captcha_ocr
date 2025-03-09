import os
import re

def rename_files(directory):
    """
    Rename files in the specified directory following the pattern:
    from: zKET_success_2c36w.png, YGqG_success_03d09.png
    to:   2c36w.png, 03d09.png
    """
    # Regular expression to match the pattern and extract the last part
    pattern = re.compile(r'.*_success_(.+\..+)')
    
    # Count successful renames
    renamed_count = 0
    error_count = 0
    
    # List all files in the directory
    files = os.listdir(directory)
    
    print(f"Found {len(files)} files in directory.")
    
    for filename in files:
        # Skip directories
        if os.path.isdir(os.path.join(directory, filename)):
            continue
            
        # Apply regex to find matching pattern
        match = pattern.match(filename)
        
        if match:
            # Extract the last part (e.g., "2c36w.png")
            new_filename = match.group(1)
            
            # Full paths for old and new files
            old_path = os.path.join(directory, filename)
            new_path = os.path.join("correct", new_filename)
            
            try:
                # Check if destination file already exists
                if os.path.exists(new_path):
                    print(f"Warning: {new_filename} already exists. Skipping {filename}")
                    error_count += 1
                    continue
                    
                # Rename the file
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {new_filename}")
                renamed_count += 1
                
            except Exception as e:
                print(f"Error renaming {filename}: {e}")
                error_count += 1
        else:
            print(f"File {filename} doesn't match the expected pattern. Skipping.")
    
    print(f"\nSummary:")
    print(f"Total files processed: {len(files)}")
    print(f"Successfully renamed: {renamed_count}")
    print(f"Errors/skipped: {error_count}")

if __name__ == "__main__":
    # Replace with your directory path
    directory_path = "correct2"
    
    # Confirm before proceeding
    print(f"This will rename files in: {directory_path}")
    confirmation = input("Do you want to continue? (y/n): ")
    
    if confirmation.lower() == 'y':
        rename_files(directory_path)
    else:
        print("Operation cancelled.")