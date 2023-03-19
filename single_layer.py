import torch


class single_layer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(single_layer, self).__init__()
        self.linear_q = torch.nn.Linear(in_features, out_features // 3)
        self.linear_k = torch.nn.Linear(in_features, out_features // 3)
        self.linear_v = torch.nn.Linear(in_features, out_features // 3)

    def update(self, target_layer):
        self.linear_q.weight.data = target_layer.weight[:target_layer.out_features // 3, :].data
        self.linear_q.bias.data = target_layer.bias[:target_layer.out_features // 3].data

        self.linear_k.weight.data = target_layer.weight[
                                    target_layer.out_features // 3:target_layer.out_features // 3 * 2, :].data
        self.linear_k.bias.data = target_layer.bias[
                                  target_layer.out_features // 3:target_layer.out_features // 3 * 2].data

        self.linear_v.weight.data = target_layer.weight[target_layer.out_features // 3 * 2:, :].data
        self.linear_v.bias.data = target_layer.bias[target_layer.out_features // 3 * 2:].data

    def forward(self, x):
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)
        return torch.concat([q, k, v], dim=-1)
