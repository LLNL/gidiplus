echo "    $@"
echo "" >> Outputs/positionalMultipleWithVariousRequired.out
echo "    $@" >> Outputs/positionalMultipleWithVariousRequired.out
$@ >> Outputs/positionalMultipleWithVariousRequired.out 2>> Outputs/positionalMultipleWithVariousRequired.err
