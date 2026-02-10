# eval_metrics.py 끝에 추가할 MRR 함수 (원본 recall_at_k 다음에 삽입)

MRR_FUNCTION = r'''
def mrr_at_k(actual, predicted, topk=50):
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
'''
