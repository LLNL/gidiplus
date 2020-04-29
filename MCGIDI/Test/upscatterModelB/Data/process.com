ROOT=~/Subversion/Fudge3/bin
$ROOT/processProtare.py -mc -mg -o neutrons -t 0.0 -t 1e-3 Original/n-001_H_001.xml n-001_H_001.xml
$ROOT/processProtare.py -mc -mg -o neutrons -t 0.0 -t 1e-3 Original/n-008_O_016.xml n-008_O_016.xml
$ROOT/processProtare.py -mc -mg -o neutrons -t 0.0 -t 1e-3 Original/n-092_U_233.xml n-092_U_233.xml
