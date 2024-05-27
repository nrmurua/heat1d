#!/bin/bash

# Define the root directory where you want to start searching
root_directory="./"

# Define the name of the file you want to search for
file_to_search="run_script.sh"

# Function to search for the file and execute the bash script
search_and_execute() {
    # Search recursively in subdirectories of the given directory
    for dirpath in $(find "$1" -mindepth 2 -maxdepth 3 -type d); do
        if [ -e "$dirpath/$file_to_search" ]; then
            echo "Executing script: $dirpath/$file_to_search"
            # Change current working directory to the directory containing the script
            cd "$dirpath"
            # Execute the bash script and wait for it to finish
            bash "$file_to_search"
            # Change back to the original directory
            cd -
        fi
    done
}

# Call the function to search and execute scripts
search_and_execute "$root_directory"


