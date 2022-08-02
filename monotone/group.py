import torch


class GroupSort(torch.nn.Module):
    def __init__(self, n_per_group: int, axis: int = -1):
        super(GroupSort, self).__init__()
        self.n_per_group = n_per_group
        self.axis = axis

    def forward(self, x: torch.Tensor):
        return group_sort(x, self.n_per_group, self.axis)

    def extra_repr(self):
        return "num_groups: {self.n_per_group}"


def get_sorting_shape(x: torch.Tensor, n_per_group: int, axis: int = -1) -> list:
    shape = list(x.shape)
    num_features = shape[axis]
    if num_features % n_per_group:
        raise ValueError(
            "number of features({num_features}) needs to be a multiple of n_per_group({n_per_group})"
        )
    shape[axis] = -1
    if axis == -1:
        shape.append(n_per_group)
    else:
        shape.insert(axis + 1, n_per_group)
    return shape


def group_sort(x: torch.Tensor, n_groups: int, axis: int = -1) -> torch.Tensor:
    if x.shape[0] == 0:
        return x
    size = get_sorting_shape(x, n_groups, axis)
    grouped_x = x.view(*size)
    sort_dim = axis if axis == -1 else axis + 1
    sorted_grouped_x, _ = grouped_x.sort(dim=sort_dim)
    sorted_x = sorted_grouped_x.view(*x.shape)
    return sorted_x
