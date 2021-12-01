echo "    $@"
echo "" >> Outputs/constructing.out
echo "    $@" >> Outputs/constructing.out
$@ >> Outputs/constructing.out 2>> Outputs/constructing.err
