
def recall_at_k_per_user(actual, pred, k):
    act_set = set(actual)
    return len(act_set & set(pred[:k])) / float(len(act_set))


def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(actual)
    true_users = 0
    for i, v in actual.items():
        act_set = set(v)
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    assert num_users == true_users
    return sum_recall / true_users


def mrr_at_k(actual, predicted, topk=50):
    """Mean Reciprocal Rank: 1/rank of first hit, averaged over users."""
    sum_rr = 0.0
    num_users = len(actual)
    for i, v in actual.items():
        act_set = set(v)
        pred_list = predicted[i][:topk]
        rr = 0.0
        for rank, item in enumerate(pred_list, 1):
            if item in act_set:
                rr = 1.0 / rank
                break
        sum_rr += rr
    return sum_rr / num_users if num_users else 0.0
