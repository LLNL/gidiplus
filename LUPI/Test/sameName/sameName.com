echo "    $@"
echo "" >> Outputs/sameName.out
echo "    $@" >> Outputs/sameName.out
$@ >> Outputs/sameName.out 2>> Outputs/sameName.err
