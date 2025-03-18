This Git repository serves as the Winter School Huckel assignment. The report is found in the root directory. The main program is found in the root directory too. 

In order to run the program, an input file must be created. It must contain at least the number of atoms:
```
n_atoms = 100
```
There are some optional values:
```
ring = False  
alternate_alpha = None
alternate_beta = None
```
The input is case sensitive, and will break if keywords are not exacly the same. Also, the values True, False and None must have the first capital letter.

In order to execute the program, the syntax is: 
```Bash
python3 huckel.py input_filename [output_filename]
```
Some input and output examples are provided in the `/examples/` directory. 


