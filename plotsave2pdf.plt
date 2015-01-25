##--------------------------------
## This script for test.
##--------------------------------
set grid

set terminal png

set output "estimation.png"
plot "result0.dat" u 1 w lines t "true(x1)", "result0.dat" u 2 w lines t "first sensor(x1)","result0.dat" u 3 w lines t "Unscented Kalman Filter(x1)"

reset