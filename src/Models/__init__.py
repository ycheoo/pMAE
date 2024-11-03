def get_model(network_global, client_id, args):
    name = args.model.lower()
    if "w_pmae" in name:
        if "l2p" in name:
            from Models.l2p_w_pmae import Learner
        elif "dualprompt" in name:
            from Models.dualprompt_w_pmae import Learner
        elif "coda_prompt" in name:
            from Models.coda_prompt_w_pmae import Learner
    elif "pmae" in name:
        from Models.pmae import Learner
    elif "l2p" in name:
        from Models.l2p import Learner
    elif "dualprompt" in name:
        from Models.dualprompt import Learner
    elif "coda_prompt" in name:
        from Models.coda_prompt import Learner
    else:
        assert 0

    return Learner(network_global, client_id, args)
