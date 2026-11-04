using NetworkInference
using Statistics
const NI = NetworkInference

function cached_context(scores)
    n = size(scores,1)
    distributions = Vector{Any}(undef,n)
    means = zeros(n); variances = zeros(n)
    for i in 1:n
        values = vcat(scores[1:i-1,i],scores[i+1:end,i])
        means[i] = mean(values); variances[i] = var(values)
        distributions[i] = try NI.fit(NI.Gamma,values) catch; nothing end
    end
    weights=zeros(n,n)
    for i in 1:n, j in i+1:n
        score=scores[i,j]
        success=false
        if distributions[i] !== nothing && distributions[j] !== nothing
            try
                weights[i,j]=NI.cdf(distributions[i],score)+NI.cdf(distributions[j],score)
                success=true
            catch
            end
        end
        if !success
            di=score-means[i]; dj=score-means[j]
            weights[i,j]=sqrt((variances[i]==0 || di<0 ? 0 : di^2/variances[i])+
                              (variances[j]==0 || dj<0 ? 0 : dj^2/variances[j]))
        end
    end
    weights
end

# Reduce directly from inferred edges: no quadratic TSV write/read round trip.
# NetworkInference emits each unordered pair once; preserve both directions.
worse(a, b) = a[1] < b[1] || (a[1] == b[1] && a[2] > b[2])

function retain!(heap, candidate, limit)
    if length(heap) < limit
        push!(heap, candidate)
        i = length(heap)
        while i > 1
            parent = div(i, 2)
            worse(heap[i], heap[parent]) || break
            heap[i], heap[parent] = heap[parent], heap[i]
            i = parent
        end
    elseif worse(heap[1], candidate)
        heap[1] = candidate
        i = 1
        while 2i <= length(heap)
            child = 2i
            if child < length(heap) && worse(heap[child+1], heap[child])
                child += 1
            end
            worse(heap[child], heap[i]) || break
            heap[i], heap[child] = heap[child], heap[i]
            i = child
        end
    end
end

function write_bounded_pairs(output_path, pairs, limit)
    targets = Dict{String, Vector{Tuple{Float64, String, Float64}}}()
    for (a, b, raw_weight) in pairs
        weight = Float64(raw_weight)
        isfinite(weight) || continue
        a == b && continue
        for (source, target) in ((a,b), (b,a))
            heap = get!(targets, target, Tuple{Float64, String, Float64}[])
            retain!(heap, (abs(weight), source, weight), limit)
        end
    end
    rows = [(source, target, weight) for (target, heap) in targets for (_,source,weight) in heap]
    sort!(rows, by = row -> (-abs(row[3]), row[1], row[2]))
    open(output_path, "w") do output
        for (source, target, weight) in rows
            println(output, source, '\t', target, '\t', weight)
        end
    end
end

# Keep the public helper for callers comparing legacy and capped output.
function write_bounded_network(output_path, network, limit)
    write_bounded_pairs(output_path,
        ((String(e.nodes[1].label), String(e.nodes[2].label), e.weight) for e in network.edges), limit)
end

function infer_weights(genes)
    scores = NI.get_puc_scores(genes, length(genes), "maximum_likelihood", 2)
    cached_context(scores)
end

function network_from_weights(genes, weights)
    n = length(genes)
    edges = [NI.Edge([genes[i], genes[j]], weights[i,j]) for i in 1:n for j in i+1:n]
    sort!(edges; by=NI.get_weight, rev=true)
    NI.InferredNetwork(genes, edges)
end

function read_nodes_preserving_labels(path)
    nodes = get_nodes(path; delim='\t')
    # readdlm's numeric inference also applies to the label column. Restore
    # literal labels one row at a time without a second full string matrix.
    open(path) do input
        readline(input)
        i = 0
        for line in eachline(input)
            isempty(strip(line)) && continue
            i += 1
            label = NI.readdlm(IOBuffer(line), '\t', String)[1]
            node = nodes[i]
            nodes[i] = NI.Node(label, node.binned_values, node.number_of_bins, node.probabilities)
        end
        i == length(nodes) || error("PIDC input label count does not match its expression rows.")
    end
    nodes
end

function main(args)
    dataset_name, output_name = args[1], args[2]
    top_k = length(args) >= 3 ? max(0, parse(Int, args[3])) : 0
    @time genes = read_nodes_preserving_labels(dataset_name)
    @time weights = infer_weights(genes)
    if top_k > 0
        # No full Edge array or global G²-edge sort for capped output.
        n = length(genes)
        pairs = ((String(genes[i].label), String(genes[j].label), weights[i,j])
                 for i in 1:n for j in i+1:n)
        write_bounded_pairs(output_name, pairs, top_k)
    else
        write_network_file(output_name, network_from_weights(genes, weights))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
