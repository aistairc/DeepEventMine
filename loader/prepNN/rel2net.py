"""Prepare relation data for networks."""


def gen_nn_rel_info(span_terms, relations, params):
    gtruth = {}
    left = []
    right = []

    for term_i, term_j in relations:
        if term_i in span_terms.term2id and term_j in span_terms.term2id:
            i = span_terms.term2id[term_i]
            j = span_terms.term2id[term_j]

            rel_id = relations[(term_i, term_j)][0]
            map_rel_type = params['mappings']['rel_map'][relations[(term_i, term_j)][1]]
            params['statistics']['rel'][map_rel_type] = params['statistics']['rel'][map_rel_type] + 1
            gtruth[i, j] = map_rel_type
            if ('_INV' not in rel_id) and (term_i != term_j):
                # if it is inverse, take the index of the element
                left.append(i)
                right.append(j)

    return gtruth, (left, right)
