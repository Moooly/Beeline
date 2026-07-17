function retain(source, target, score,    count_now, slot, key, cursor, weakest, weakest_score) {
    if (!(target in target_seen)) {
        target_seen[target] = 1
        target_order[++target_total] = target
    }
    count_now = target_count[target]
    if (count_now < top_k) {
        slot = count_now + 1
        target_count[target] = slot
    } else {
        weakest = 1
        weakest_score = edge_score[target SUBSEP 1]
        for (cursor = 2; cursor <= count_now; cursor++) {
            key = target SUBSEP cursor
            if (edge_score[key] < weakest_score ||
                (edge_score[key] == weakest_score && edge_sequence[key] < edge_sequence[target SUBSEP weakest])) {
                weakest = cursor
                weakest_score = edge_score[key]
            }
        }
        if (score <= weakest_score) return
        slot = weakest
    }
    key = target SUBSEP slot
    edge_source[key] = source
    edge_score[key] = score
    edge_sequence[key] = sequence
}

BEGIN {
    FS = ","
    OFS = "\t"
    while ((getline gene < gene_file) > 0) genes[++gene_count] = gene
    close(gene_file)
    total = gene_count * gene_count
    sequence = 0
}

{
    row = NR
    for (column = 1; column <= NF; column++) {
        sequence++
        retain(genes[row], genes[column], total - ($column + 0))
    }
}

END {
    print "Gene1", "Gene2", "EdgeWeight"
    for (target_index = 1; target_index <= target_total; target_index++) {
        target = target_order[target_index]
        count_now = target_count[target]
        for (rank = 1; rank <= count_now; rank++) {
            selected = 0
            for (slot = 1; slot <= count_now; slot++) {
                key = target SUBSEP slot
                emitted_key = key SUBSEP "emitted"
                if (emitted[emitted_key]) continue
                if (selected == 0 || edge_score[key] > edge_score[selected_key] ||
                    (edge_score[key] == edge_score[selected_key] && edge_sequence[key] < edge_sequence[selected_key])) {
                    selected = slot
                    selected_key = key
                }
            }
            emitted[selected_key SUBSEP "emitted"] = 1
            print edge_source[selected_key], target, sprintf("%.17g", edge_score[selected_key])
        }
    }
}
