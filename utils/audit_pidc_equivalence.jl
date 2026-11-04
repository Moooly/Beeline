include(ARGS[1])
using Random, LinearAlgebra
quiet = open("/dev/null", "w")
for n in [2, 10, 100]
    Random.seed!(42)
    a = rand(n,n); scores = a+a'; scores[diagind(scores)] .= 0
    nodes = [NI.Node(string(i), [], 0, []) for i in 1:n]
    for degenerate in [false, true]
        if degenerate scores[1,:] .= 0; scores[:,1] .= 0 end
        reference = redirect_stdout(quiet) do
            NI.get_weights(NI.PIDCNetworkInference(), scores, n, nodes)
        end
        @assert isequal(Array(reference), cached_context(scores))
    end
end
println("context fallback/degenerate comparisons exact")
for path in ARGS[2:end]
    genes = read_nodes_preserving_labels(path); n = length(genes)
    ref = redirect_stdout(quiet) do
        NI.InferredNetwork(NI.PIDCNetworkInference(), genes)
    end
    weights = infer_weights(genes)
    actual = network_from_weights(genes, weights)
    rows(net) = [(e.nodes[1].label, e.nodes[2].label, e.weight) for e in net.edges]
    @assert isequal(rows(ref), rows(actual))
    # Independent full-sort oracle for cap and boundary ties.
    full = [(String(a), String(b), Float64(e.weight)) for e in ref.edges
            for (a,b) in [(e.nodes[1].label,e.nodes[2].label),(e.nodes[2].label,e.nodes[1].label)]
            if a != b && isfinite(e.weight)]
    order(row) = (-abs(row[3]), row[1], row[2])
    for k in [1, 5, n+1]
        expected = Tuple{String,String,Float64}[]
        for target in unique([r[2] for r in full])
            group = sort(filter(r -> r[2] == target, full); by=order)
            append!(expected,group[1:min(k,length(group))])
        end
        sort!(expected;by=order)
        out = tempname()
        write_bounded_pairs(out, ((String(genes[i].label), String(genes[j].label), weights[i,j]) for i in 1:n for j in i+1:n), k)
        expected_text = join([string(r[1],'\t',r[2],'\t',r[3],'\n') for r in expected])
        @assert read(out,String) == expected_text
        rm(out)
    end
    old=Float64[]; new=Float64[]
    for iteration in 1:3
        push!(old,@elapsed redirect_stdout(quiet) do
            NI.InferredNetwork(NI.PIDCNetworkInference(),genes)
        end)
        push!(new,@elapsed network_from_weights(genes,infer_weights(genes)))
    end
    println("genes=",n," full/capped exact old=",median(old)," new=",median(new))
end
