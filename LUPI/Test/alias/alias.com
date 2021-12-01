echo "    $@"
echo "" >> Outputs/alias.out
echo "    $@" >> Outputs/alias.out
$@ >> Outputs/alias.out 2>> Outputs/alias.err
