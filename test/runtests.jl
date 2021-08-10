using Revise, Test 

P = Dict(:length=>10,:width=>7, :Barrier_height=>1.0,:SCpx_magnitude=>0.4,:SCDW_position=>0.3,) 
P = Dict(:length=>10,:width=>7, :Barrier_height=>1.0,:SCpx_magnitude=>0.4,:SCDW_position=>0.3,:delta=>0.002,:AtomToLayer=>"forced")
using Constants: PATH_SNAKE 

include("$PATH_SNAKE/input_file.jl")

#include("h.jl")

println() 

include("gf.jl")

println() 

include("plot.jl")





































































nothing
