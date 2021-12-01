tail -n +2 $2 > file1
tail -n +2 $3 > file2
diff file1 file2 > /dev/null
if [ $? != 0 ]; then echo "FAILURE: $1": difference in output; fi
rm -f file1 file2
