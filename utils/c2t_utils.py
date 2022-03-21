import torch

# C2T: Using to padding
MAX_NESTED = 4
PAIR_SIZE = 2

# Type padding
TYPE_TRID = 0
TYPE_ARG1 = 1
TYPE_ARG2 = 2
TYPE_LBL = 4
TYPE_TR0ID = 5


def _is_contain(input, target):
    """ C2T: Check and return the index if Tensor target(list) contains Tensor input
    """
    for i, e in enumerate(target):
        if torch.all(torch.eq(e, input)):
            return i
    return -1


# C2T: Padding
# C2T
def _truncate(arr, max_length):
    while True:
        total_length = len(arr)
        if total_length <= max_length:
            break
        else:
            arr.pop()


def _padding(arr, max_length, padding_idx=-1):
    while len(arr) < max_length:
        arr.append(padding_idx)


def _to_tensor(arr, params):
    return torch.tensor(arr, device=params['device'])


def _to_torch_data(arr, max_length, params, padding_idx=-1):
    for e in arr:
        _truncate(e, max_length)
        _padding(e, max_length, padding_idx=padding_idx)
    return _to_tensor(arr, params)


def _padding_rels(rels, max_rel_per_event):
    """ C2T: Padding relations
    """
    padded_rels = []
    for rel in rels:
        padded_rels.append(rel)
    while len(padded_rels) < max_rel_per_event:
        padded_rels.append([-1] * PAIR_SIZE)
    return padded_rels


def _padding_cell_1_value(val, cols, rows, padding_val=-1):
    """ C2T: Padding cells that only have 1 value
        """
    padded_cell = []
    padded_row = []
    padded_row.append(val)
    while len(padded_row) < cols:
        padded_row.append(padding_val)
    padded_cell.append(padded_row)
    while len(padded_cell) < rows:
        padded_cell.append([padding_val] * cols)
    return padded_cell


def _padding_even(even, max_rel_per_event):
    max_cell = 4
    padding_val = -1
    padded_even = []
    # padding trid
    trid = even[0]
    padded_trid = [_padding_cell_1_value(trid[0], PAIR_SIZE, max_rel_per_event, padding_val=padding_val),
                   _padding_cell_1_value(trid[1], PAIR_SIZE, max_rel_per_event, padding_val=padding_val)]
    while len(padded_trid) < max_cell:
        padded_trid.append(_padding_cell_1_value(padding_val, PAIR_SIZE, max_rel_per_event, padding_val=padding_val))
    # print('padded_trid', padded_trid)
    padded_even.append(padded_trid)
    # padding arg1
    arg1 = even[1]
    padded_arg1 = []
    for e in arg1:
        padded_arg1.append(_padding_cell_1_value(e, PAIR_SIZE, max_rel_per_event, padding_val=padding_val))
    # print('padded_arg1', padded_arg1)
    padded_even.append(padded_arg1)
    # padding arg2
    arg2 = even[2]
    padded_arg2 = []
    for e in arg2:
        padded_arg2.append(_padding_cell_1_value(e, PAIR_SIZE, max_rel_per_event, padding_val=padding_val))
    while len(padded_arg2) < max_cell:
        padded_arg2.append(_padding_cell_1_value(padding_val, PAIR_SIZE, max_rel_per_event, padding_val=padding_val))
    # print('padded_arg2', padded_arg2)
    padded_even.append(padded_arg2)
    # padding r
    r = even[3]
    padded_r = []
    for e in r:
        if e != -1:
            padded_r.append(_padding_rels([e], max_rel_per_event))
        else:
            padded_r.append(_padding_cell_1_value(padding_val, PAIR_SIZE, max_rel_per_event, padding_val=padding_val))
    while len(padded_r) < max_cell:
        padded_r.append(_padding_cell_1_value(padding_val, PAIR_SIZE, max_rel_per_event, padding_val=padding_val))
    padded_even.append(padded_r)
    # padding label
    lbl = even[4]
    padded_lbl = [_padding_cell_1_value(lbl, PAIR_SIZE, max_rel_per_event, padding_val=padding_val)]
    while len(padded_lbl) < max_cell:
        padded_lbl.append(_padding_cell_1_value(padding_val, PAIR_SIZE, max_rel_per_event, padding_val=padding_val))
    padded_even.append(padded_lbl)

    # padding tr0id
    if len(even) > 5:
        tr0id = even[5]
        padded_tr0id = []
        for e in tr0id:
            padded_tr0id.append(_padding_cell_1_value(e, PAIR_SIZE, max_rel_per_event, padding_val=padding_val))
        while len(padded_tr0id) < max_cell:
            padded_tr0id.append(
                _padding_cell_1_value(padding_val, PAIR_SIZE, max_rel_per_event, padding_val=padding_val))
        padded_even.append(padded_tr0id)

    return padded_even


def _padding_truth_ev(arr, max_rel_per_event, max_ev_per_layer, max_ev_per_batch):
    padded_arr = []
    for row in arr:
        padded_row = []
        for cell in row:
            padded_cell = []
            if cell != -1:
                for ev in cell:
                    r_e = ev[0]
                    e_e = ev[1]
                    padded_r_e = []
                    for r_e_e in r_e:
                        while len(r_e_e) < max_rel_per_event:
                            r_e_e = r_e_e + [-1]
                        padded_r_e.append(r_e_e)
                    while len(padded_r_e) < max_rel_per_event:
                        padded_r_e.append([-1] * max_rel_per_event)
                    while len(e_e) < max_rel_per_event:
                        e_e = e_e + [-1]
                    padded_r_e.append(e_e)
                    padded_cell.append(padded_r_e)
                while len(padded_cell) < max_ev_per_layer:
                    padded_cell.append([[-1] * max_rel_per_event] * (max_rel_per_event + 1))
            else:
                padded_cell = [[[-1] * max_rel_per_event] * (max_rel_per_event + 1)] * max_ev_per_layer
            padded_row.append(padded_cell)
        padded_arr.append(padded_row)
    while len(padded_arr) < max_ev_per_batch:
        padded_arr.append([[[[-1] * max_rel_per_event] * (max_rel_per_event + 1)] * max_ev_per_layer] * MAX_NESTED)
    return padded_arr


def _flatten_structs_type_ev(arr, max_rel_per_event, max_ev_per_tr):
    padded_arr = []
    for cell in arr:
        padded_cell = []
        if cell != -1:
            for ev in cell:
                padded_r_e = []
                for r_e_e in ev:
                    # while len(r_e_e) < PAIR_SIZE:
                    #     r_e_e = r_e_e + [-1]
                    padded_r_e.append(r_e_e)
                while len(padded_r_e) < max_rel_per_event:
                    padded_r_e.append([-1] * PAIR_SIZE)
                # if len(padded_r_e) > 2:
                padded_cell.append(padded_r_e)
            while len(padded_cell) < max_ev_per_tr:
                padded_cell.append([[-1] * PAIR_SIZE] * max_rel_per_event)
        else:
            padded_cell = [[[-1] * PAIR_SIZE] * max_rel_per_event] * max_ev_per_tr
        padded_arr.append(padded_cell)
    return padded_arr


def _padding_even_cd(even_cd, max_rel_per_event, dtype, device=torch.device("cpu")):
    padded_even = []
    r_part = even_cd[0]
    e_part = even_cd[1]
    for r in r_part:
        padded_r = []
        for r_e in r:
            padded_r.append(r_e)
        while len(padded_r) < max_rel_per_event:
            padded_r.append(-1)
        padded_even.append(padded_r)
    while len(padded_even) < max_rel_per_event:
        padded_even.append([-1] * max_rel_per_event)
    padded_e = []
    for e in e_part:
        padded_e.append(e)
    while len(padded_e) < max_rel_per_event:
        padded_e.append(-1)
    padded_even.append(padded_e)
    return torch.tensor(padded_even, dtype=dtype, device=device)


# C2T: Un-padding
def _unpadding_cell_1_value(padded_cell, cols, rows, device, padding_val=-1, replacing_padding=-1):
    padding_cell = torch.tensor([[padding_val] * cols] * rows, device=device)
    if torch.all(torch.eq(padded_cell.long(), padding_cell)):
        return replacing_padding
    else:
        return padded_cell[0][0]


def _unpadding_even_element(padded_ev_e, max_rel_per_event, device, type_padding=0, replacing_padding=-1):
    unpadded_ev_e = []
    padding_val = -1
    if type_padding == TYPE_TRID:
        valid_idx = 1
    elif type_padding == TYPE_ARG1:
        valid_idx = 3
    elif type_padding == TYPE_ARG2:
        valid_idx = 2
    elif type_padding == TYPE_LBL:
        valid_idx = 0
    elif type_padding == TYPE_TR0ID:
        valid_idx = 2

    for e in padded_ev_e:
        unpadded_e = []
        for i, cell in enumerate(e):
            cell = cell.to(device)
            if i <= valid_idx:
                unpadded_e.append(
                    _unpadding_cell_1_value(cell, PAIR_SIZE, max_rel_per_event, device, padding_val=padding_val,
                                            replacing_padding=replacing_padding))
        unpadded_ev_e.append(unpadded_e)
    return torch.tensor(unpadded_ev_e, device=device)
