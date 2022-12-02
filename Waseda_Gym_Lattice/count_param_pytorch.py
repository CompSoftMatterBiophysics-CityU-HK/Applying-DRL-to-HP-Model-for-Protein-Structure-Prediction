from prettytable import PrettyTable

def count_parameters(model):
    """
    # for sum the number of elements for every parameter group
    sum(p.numel() for p in model.parameters())

    # calculate only the trainable parameters
    sum(p.numel() for p in model.parameters() if p.requires_grad)

    from https://newbedev.com/check-the-total-number-of-parameters-in-a-pytorch-model
    and https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
    """
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
