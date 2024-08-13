#!/bin/bash

root_directory="./"

file_to_search="run_script.sh"

search_and_execute() {
    for dirpath in $(find "$1" -mindepth 2 -maxdepth 3 -type d); do
        if [ -e "$dirpath/$file_to_search" ]; then
            echo "Executing script: $dirpath/$file_to_search"
            cd "$dirpath"
            bash "$file_to_search"
            cd -
        fi
    done
}

search_and_execute "$root_directory"


