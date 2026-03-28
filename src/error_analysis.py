def sample_errors(preds, refs, n=25):
    errors = [(p, r) for p, r in zip(preds, refs) if p != r]

    step = max(1, len(errors)//n)

    return errors[::step][:n]