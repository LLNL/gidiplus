echo "    $@"
echo "" >> Outputs/missingValue1.out
echo "    $@" >> Outputs/missingValue1.out
$@ >> Outputs/missingValue1.out 2>> Outputs/missingValue1.err
