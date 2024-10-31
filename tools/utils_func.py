import torch

def block_influence(
    input_hidden_state: torch.Tensor,
    output_hidden_state: torch.Tensor,
    angular=False,
):
    """
    input_hidden_state: B, S, D
    output_hidden_state: B, S, D
    """
    _, _, d = input_hidden_state.shape
    input_hidden_state = input_hidden_state.reshape(-1, d)
    output_hidden_state = output_hidden_state.reshape(-1, d)

    norm_input = input_hidden_state.norm(dim=-1, keepdim=True)
    norm_output = output_hidden_state.norm(dim=-1, keepdim=True)

    sim = (input_hidden_state @ output_hidden_state.T) / (norm_input * norm_output)
    sim = sim.diagonal().nan_to_num(nan=0.5)

    if angular:
        return (torch.arccos(sim) / torch.pi)

    return 1 - sim


def jaccard_similarity(list1, list2):
    if isinstance(list1, list):
        list1 = set(list1)
    
    if isinstance(list2, list):
        list2 = set(list2)
    intersection = list1.intersection(list2)
    union = list1.union(list2)

    if len(union) > 0:
        overlap_ratio = len(intersection) / len(union)
    else:
        overlap_ratio = 0

    return overlap_ratio


def adaptive_rank_selection(svd_importance_list, target_ratio):
    total_sum = sum(svd_importance_list)
    target_sum = total_sum * target_ratio

    sorted_list = sorted(enumerate(svd_importance_list), key=lambda x: -x[1])

    cumulative_sum = 0
    indices = []
    for index, value in sorted_list:
        cumulative_sum += value
        indices.append(index)
        if cumulative_sum >= target_sum:
            break
    return indices