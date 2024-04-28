#!/usr/bin/sh
TEMP_FILE="./temp"


succuess () {
   echo -e "$1:\e[32m Succussful \e[0m"
}

failure () {
   echo -e "${test_file}:\e[31m Failed \e[0m"
}

# Change to the directory containting this script
cd "$(dirname "$0")"

for test_file in $(ls ./tests/test*.cel); do
    test_name="${test_file##*/}"
    test_name="${test_name%.*}"

    test_res="./tests/results/${test_name}"
    [ ! -f "$test_res" ] && echo -e "\e[33mTest result for ${test_file} doesn't exist\e[0m" && continue

    ./celestine/main.py -r "$test_file" > "$TEMP_FILE" 2>&1
    if [[ "$?" == 0 ]]; then rm "./tests/${test_name}.out"; fi

    if cmp -- "$TEMP_FILE" "$test_res"; then
        succuess "$test_file"
    else
        failure "$test_file"
    fi
done

for test_file in $(ls ./tests/test*.py); do
    ${test_file}

    if [[ "$?" == 0 ]]; then
        succuess "$test_file"
    else
        failure "$test_file"
    fi
done

rm $TEMP_FILE