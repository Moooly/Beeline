function absolute(value) { return value < 0 ? -value : value }

function retain(source, target, score_text,    count_now, slot, key, cursor, weakest, weakest_abs) {
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
        weakest_abs = edge_abs[target SUBSEP 1]
        for (cursor = 2; cursor <= count_now; cursor++) {
            key = target SUBSEP cursor
            if (edge_abs[key] < weakest_abs ||
                (edge_abs[key] == weakest_abs && edge_sequence[key] < edge_sequence[target SUBSEP weakest])) {
                weakest = cursor
                weakest_abs = edge_abs[key]
            }
        }
        if (absolute(score_text + 0) <= weakest_abs) return
        slot = weakest
    }
    key = target SUBSEP slot
    edge_source[key] = source
    edge_score[key] = score_text
    edge_abs[key] = absolute(score_text + 0)
    edge_sequence[key] = sequence
}

BEGIN { OFS = "\t"; sequence = 0 }

NR == 1 && input_header { next }

{
    sequence++
    retain($source_col, $target_col, $score_col)
}

END {
    print output_source, output_target, output_score
    for (target_index = 1; target_index <= target_total; target_index++) {
        target = target_order[target_index]
        count_now = target_count[target]
        for (rank = 1; rank <= count_now; rank++) {
            selected = 0
            for (slot = 1; slot <= count_now; slot++) {
                key = target SUBSEP slot
                emitted_key = key SUBSEP "emitted"
                if (emitted[emitted_key]) continue
                if (selected == 0 || edge_abs[key] > edge_abs[selected_key] ||
                    (edge_abs[key] == edge_abs[selected_key] && edge_sequence[key] < edge_sequence[selected_key])) {
                    selected = slot
                    selected_key = key
                }
            }
            emitted[selected_key SUBSEP "emitted"] = 1
            print edge_source[selected_key], target, edge_score[selected_key]
        }
    }
}
