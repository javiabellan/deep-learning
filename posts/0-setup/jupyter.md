# Jupyter notebook


## Basics

* `Ctrl`+`Enter`: Run cell
* `Shift`+`Enter`: Run cell and next



## Run bash commands
with `!`


## Show documentation
* Place the cursor in a function and type `Shift`+`Tab`: For showing signature and docstring (Press more tabs for mor info)
* `?some_expression` for showing docstring
* `??some_expression` for showing actial code

## Python debugger

You can use the python debugger `pdb` to step through code. You can use it in 2 ways:

1. Go tou your code and write `pdb.set_trace()` to set a *breakpoint*.
2. `%debug` to trace an error

Inside the python debugger (a command prompt) you can write python expressions and debbug commands:

* `s`: **Step** into subroutines.
* `n`: Execute this line (`->`) until the **next** line is reached (or just press enter).
* `c`: **Continue** execution. Means stop debugging or reach next breakpoint.
* `u`: Go one level **up** in the stack trace (no execution here).
* `d`: Go one level **down** in the stack trace (no execution here).
* `p`: **print** the value of some variable.
* `l`: **list** (show) a few of lines of code before and after.
