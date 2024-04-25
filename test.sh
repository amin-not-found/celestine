#!/usr/bin/sh
TESTS_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )/tests"
TEMP_FILE="${TESTS_DIR}/temp"


succuess () {
   echo -e "$1:\e[32m Succussful \e[0m"
}

failure () {
   echo -e "${test_file}:\e[31m Failed \e[0m"
}


for test_file in $(ls ${TESTS_DIR}/test*.cel); do
    test_name="${test_file##*/}"
    test_name="${test_name%.*}"

    test_res="${TESTS_DIR}/results/${test_name}"
    [ ! -f "$test_res" ] && echo -e "\e[33mTest result for ${test_file} doesn't exist\e[0m" && continue

    ${TESTS_DIR}/../celestine/main.py -r "$test_file" > "$TEMP_FILE"
    rm "${TESTS_DIR}/${test_name}.out"

    if cmp -- "$TEMP_FILE" "$test_res"; then
        succuess "$test_file"
    else
        failure "$test_file"
    fi
done

for test_file in $(ls ${TESTS_DIR}/test*.py); do
    ${test_file}

    if [[ "$?" == 0 ]]; then
        succuess "$test_file"
    else
        failure "$test_file"
    fi
done

rm $TEMP_FILE