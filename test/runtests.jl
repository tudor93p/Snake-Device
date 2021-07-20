using Revise, Test 

P = Dict(:length=>10,:width=>7, :Barrier_height=>1.0,:SCpx_magnitude=>0.4,:SCDW_position=>0.3,) 
P = Dict(:length=>10,:width=>7, :Barrier_height=>1.0,:SCpx_magnitude=>0.4,:SCDW_position=>0.3,:delta=>0.002,:AtomToLayer=>"forced")


input_GF =  Dict(:allparams=>(

										delta = [0.07], 
										), 

									 :digits=>(
											delta = (1,3),
										)	
									
									)


input_dict = Dict(:allparams=>(
										length = [10,20],
									 	width = [8],
										Barrier_height = [0,0.5],
										SCpx_magnitude = [0.6],
										),

									 :digits=>(
											length = (3, 0),

											Barrier_height = (1,3),

											SCpx_magnitude = (1,3),

										)	
									
									)


#include("h.jl")

println() 

#include("gf.jl")

println() 

include("plot.jl")





































































nothing