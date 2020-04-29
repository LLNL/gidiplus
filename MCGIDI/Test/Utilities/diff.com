diff $2 $3 > /dev/null
if [ $? != 0 ]; then echo "FAILURE: $1": difference in output; fi
