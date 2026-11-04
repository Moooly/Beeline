# Indexed min-heap per target. O(log K) updates; O(targets * K) storage.
function absolute(value) { return value < 0 ? -value : value }
function worse(a, b) {
    return edge_abs[a] < edge_abs[b] || (edge_abs[a] == edge_abs[b] && ("x" edge_source[a]) > ("x" edge_source[b]))
}
function swap(target, a, b,    ka,kb,tmp) {
    ka=target SUBSEP a; kb=target SUBSEP b
    tmp=edge_source[ka]; edge_source[ka]=edge_source[kb]; edge_source[kb]=tmp
    tmp=edge_score[ka]; edge_score[ka]=edge_score[kb]; edge_score[kb]=tmp
    tmp=edge_abs[ka]; edge_abs[ka]=edge_abs[kb]; edge_abs[kb]=tmp
    position[target SUBSEP edge_source[ka]]=a; position[target SUBSEP edge_source[kb]]=b
}
function down(target, slot,    child,count_now) {
    count_now=target_count[target]
    while (slot*2 <= count_now) {
        child=slot*2
        if (child < count_now && worse(target SUBSEP (child+1), target SUBSEP child)) child++
        if (!worse(target SUBSEP child, target SUBSEP slot)) break
        swap(target,slot,child); slot=child
    }
}
function retain(source, target, score_text,    key,slot,parent,score,root) {
    if (source == target || source == "" || target == "") return
    if (score_text !~ /^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$/) return
    score=score_text+0
    if (!(target in target_seen)) { target_seen[target]=1; target_order[++target_total]=target }
    key=target SUBSEP source
    slot=position[key]
    if (slot) {
        if (absolute(score) < edge_abs[target SUBSEP slot] ||
            (absolute(score) == edge_abs[target SUBSEP slot] && score <= edge_score[target SUBSEP slot]+0)) return
        edge_score[target SUBSEP slot]=score_text; edge_abs[target SUBSEP slot]=absolute(score)
        down(target,slot); return
    }
    if (target_count[target] >= top_k) {
        root=target SUBSEP 1
        if (absolute(score) < edge_abs[root] ||
            (absolute(score) == edge_abs[root] && ("x" source) >= ("x" edge_source[root]))) return
        delete position[target SUBSEP edge_source[root]]
        slot=1
    } else slot=++target_count[target]
    key=target SUBSEP slot
    edge_source[key]=source; edge_score[key]=score_text; edge_abs[key]=absolute(score)
    position[target SUBSEP source]=slot
    if (slot == 1) down(target,slot)
    else while (slot > 1) {
        parent=int(slot/2)
        if (!worse(target SUBSEP slot, target SUBSEP parent)) break
        swap(target,slot,parent); slot=parent
    }
}
function emit(target,    count_now,rank,key) {
    count_now=target_count[target]
    # Extract worst first, then emit in descending rank without a quadratic scan.
    for (rank=count_now;rank>=1;rank--) {
        key=target SUBSEP 1
        sources[rank]=edge_source[key]; scores[rank]=edge_score[key]
        swap(target,1,rank); target_count[target]--; down(target,1)
    }
    for (rank=1;rank<=count_now;rank++) print sources[rank],target,scores[rank]
}

BEGIN { OFS = "\t" }
NR == 1 && input_header {
    if (strict_grnvbem && $0 !~ /^Parent[ \t\r]+(-|Type)[ \t\r]+Child[ \t\r]+Weight[ \t\r]+Probability[ \t\r]+Score[ \t\r]*$/) {
        invalid=1; exit 1
    }
    next
}
{
    if (strict_grnvbem && (NF != 6 || $source_col == "" || $target_col == "" ||
        $score_col !~ /^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$/ ||
        sprintf("%.17g", $score_col+0) !~ /^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$/)) {
        invalid=1; exit 1
    }
    retain($source_col, $target_col, $score_col)
}
END {
    if (invalid || (strict_grnvbem && NR == 0)) {
        print "Invalid GRNVBEM output near row " NR ": expected a header and finite six-column edge rows." > "/dev/stderr"
        exit 1
    }
    print output_source, output_target, output_score
    for (i=1;i<=target_total;i++) emit(target_order[i])
}
