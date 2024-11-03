def get_fed_method(args):
    name = args.method.lower()
    if name == "fedavg":
        from Methods.FedAVG import Method
    elif "mae" in name:
        from Methods.pMAE import Method
    else:
        assert 0

    return Method(args)
