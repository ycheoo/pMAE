def get_local(args):
    name = args.method.lower()
    if "pmae" in name:
        from Locals.pMAELocal import Local
    else:
        from Locals.BaseLocal import Local

    return Local(args)
