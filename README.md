# Saturation-Calculators
Code for calculating the severity of saturation effects for atmospheric
 resonance lidars

The subfolders contain the python code for computing the degree of saturation 
for calcium and calcium plus resonance lidars, for sodium, iron, and helium 
resonance lidars, and for potassium resonance lidars, respectively, as
presented in the corresponding publications (DOIs to be added when available).
In each case, there is a file containing the library of functions needed for
the saturation calculations and a file containing the code necessary to 
reproduce the plots presented in the respective publications. The code 
libraries for each resonance lidar target are distinct, due to particularities
such as the hyperfine structure or existence of multiple isotopes, though there
is significant overlap.

Mentions in the code of 'VDG' refer to the approach to calculating saturation
effects presented in the paper "Saturation Effects in Na Lidar Measurements" by
Peter von der Gathen (1991, https://doi.org/10.1029/90JA02420). Mentions of
'Megie' refer to the approximation of saturation effects derived by Megie et
 al. in the paper "Simultaneous nighttime Lidar measurements of atmospheric 
 sodium and potassium" (1978, https://doi.org/10.1016/0032-0633(78)90034-X).
