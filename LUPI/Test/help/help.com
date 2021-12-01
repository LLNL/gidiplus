echo "    $@"
echo "" >> Outputs/help.out
echo "    $@" >> Outputs/help.out
$@ >> Outputs/help.out 2>> Outputs/help.err
