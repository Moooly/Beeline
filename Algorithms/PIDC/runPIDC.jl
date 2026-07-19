# Include packages

using NetworkInference
using LightGraphs

algorithm = PIDCNetworkInference()

dataset_name = string(ARGS[1])
output_name = string(ARGS[2])
top_k = length(ARGS) >= 3 ? max(0, parse(Int, ARGS[3])) : 0

function compact_network_file(input_path, output_path, limit)
    target_edges = Dict{String, Vector{Tuple{Float64, Int, String}}}()
    sequence = 0
    for line in eachline(input_path)
        fields = split(chomp(line), '\t')
        length(fields) < 3 && continue
        score = tryparse(Float64, fields[3])
        score === nothing && continue
        sequence += 1
        target = String(fields[2])
        candidate = (abs(score), sequence, line)
        heap = get!(target_edges, target, Tuple{Float64, Int, String}[])
        if length(heap) < limit
            push!(heap, candidate)
            continue
        end
        weakest_index = argmin([(edge[1], edge[2]) for edge in heap])
        if candidate[1] > heap[weakest_index][1]
            heap[weakest_index] = candidate
        end
    end

    open(output_path, "w") do output
        for heap in values(target_edges)
            sort!(heap, by=edge -> (-edge[1], edge[2]))
            for edge in heap
                println(output, edge[3])
            end
        end
    end
end

@time genes = get_nodes(dataset_name; delim='\t');

@time network = InferredNetwork(algorithm, genes);

if top_k > 0
    temporary_output = tempname()
    try
        write_network_file(temporary_output, network)
        compact_network_file(temporary_output, output_name, top_k)
    finally
        isfile(temporary_output) && rm(temporary_output)
    end
else
    write_network_file(output_name, network)
end
