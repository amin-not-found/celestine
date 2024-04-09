#!/usr/bin/sh
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
TEMP_FILE="temp"

tests=($(ls test*.cel))
echo "Test files: ${tests[@]}"

for test_file in ${tests[@]}; do
    test_res="${test_file%.*}"
    [ ! -f "$test_res" ] && echo -e "\e[33mTest result for ${test_file} doesn't exist\e[0m" && continue

    ${SCRIPT_DIR}/../main.py -r "$test_file" > "$TEMP_FILE"
    rm "${test_res}.out"

    if cmp -- "$TEMP_FILE" "$test_res"; then
        echo -e "${test_file}:\e[32m Succussful \e[0m"
    else
        echo -e "${test_file}:\e[31m Failed \e[0m"
    fi
done

rm temp