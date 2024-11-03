def get_server(args):
    name = args.method.lower()
    if "pmae" in name:
        from Servers.pMAEServer import Server
    else:
        from Servers.BaseServer import Server

    return Server(args)
