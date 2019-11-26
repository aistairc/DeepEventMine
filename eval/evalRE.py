import torch


class SelectClass:
    """
        Correct predictions: From 2 direction relations choose
        - if both 'no_relation' -> no_relation
        - if one positive, one negative -> positive
        - if both positive -> more confident (highest probability)
    """

    def __init__(self, params):
        self.params = params

    # Without L R distinguishing & on CPU
    def __call__(self, *inputs):
        labmap = torch.tensor(
            [self.params['lab_map'][e] for e in range(0, self.params['voc_sizes']['rel_size'])])
        ignore = torch.tensor(self.params['lab2ign_id'])

        cpu_device = torch.device("cpu")
        y_lr, y_rl = inputs
        y_lr = y_lr.to(cpu_device)
        y_rl = y_rl.to(cpu_device)
        if self.params['fp16']:
            y_lr = y_lr.float()
            y_rl = y_rl.float()

        labels_lr = y_lr.argmax(dim=1).view(-1)
        labels_rl = y_rl.argmax(dim=1).view(-1)

        m = torch.arange(labels_lr.shape[0])

        lr_probs = y_lr[m, labels_lr]
        rl_probs = y_rl[m, labels_rl]
        inv_lr = labmap[labels_lr]
        inv_rl = labmap[labels_rl]

        negative_val = torch.tensor(-1)

        # if both are negative --> keep negative class as prediction (1:Other:2)
        a_x1 = torch.where((labels_lr == labels_rl) & (labels_lr == ignore), ignore, negative_val)

        # if both are positive with same label (e.g. 1:rel:2) --> choose from probability
        a4 = torch.where((labels_lr != ignore) & (labels_rl != ignore) & (labels_lr == labels_rl),
                         lr_probs, negative_val.float())
        a5 = torch.where((labels_lr != ignore) & (labels_rl != ignore) & (labels_lr == labels_rl),
                         rl_probs, negative_val.float())
        a_x4 = torch.where((a4 >= a5) & (a4 != -1) & (a5 != -1), labels_lr, negative_val)
        a_x5 = torch.where((a4 < a5) & (a4 != -1) & (a5 != -1), inv_rl, negative_val)

        # # if both are positive with inverse 1:rel:2 & 2:rel:1 (this is correct) --> keep them the 'rel' label
        a_x6 = torch.where((labels_lr != labels_rl) & (labels_lr != ignore) &
                           (labels_rl != ignore) & (inv_lr == labels_rl), labels_lr, negative_val)

        # if one positive & one negative --> choose the positive class
        a_x2 = torch.where((labels_lr != labels_rl) & (labels_lr == ignore) & (labels_rl != ignore),
                           inv_rl, negative_val)
        a_x3 = torch.where((labels_lr != labels_rl) & (labels_lr != ignore) & (labels_rl == ignore),
                           labels_lr, negative_val)

        # if both are positive with different labels --> choose from probability
        a7 = torch.where(
            (labels_lr != ignore) & (labels_rl != ignore) & (labels_lr != labels_rl) & (inv_lr != labels_rl),
            lr_probs, negative_val.float())
        a8 = torch.where(
            (labels_lr != ignore) & (labels_rl != ignore) & (labels_lr != labels_rl) & (inv_lr != labels_rl), rl_probs,
            negative_val.float())

        a_x7 = torch.where((a7 >= a8) & (a7 != -1) & (a8 != -1), labels_lr, negative_val)
        a_x8 = torch.where((a7 < a8) & (a7 != -1) & (a8 != -1), inv_rl, negative_val)

        fin = torch.stack([a_x1, a_x2, a_x3, a_x4, a_x5, a_x6, a_x7, a_x8])
        assert (torch.sum(torch.clamp(fin, min=-1.0, max=0.0), dim=0) == -7).all(), "check evaluation"
        fin_preds = torch.max(fin, dim=0)

        return fin_preds[0]


def calc_stats(preds, params):
    new_preds = SelectClass(params)(preds[0], preds[1])
    return new_preds
