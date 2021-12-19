using Revise, Test 
using Constants: PATH_SNAKE 
include("$PATH_SNAKE/input_file.jl")

import Device 



for f in (
#
#"h",
#
#"gf",
#


"ribbon",
#"plot",

#pr_in("ribbon_FS")


#pr_in("analytical")




)

	println("\n********* $f ********* \n")

	include("$f.jl")

end





































































nothing
