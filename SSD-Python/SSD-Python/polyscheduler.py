import numpy as np
def polySchedulerLR(base_value, epochs, niter_per_ep, warmup_epochs = 0):
    max_iters = epochs * niter_per_ep
    warmup_iters = warmup_epochs * niter_per_ep
    iters = np.arange(max_iters - warmup_iters)
    
    warmup_schedule = np.array([])
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(0, base_value, warmup_iters)
    schedule = base_value * (1.0 - iters/ max_iters) ** 0.9
    schedule = np.concatenate(warmup_schedule, schedule)
    
    assert len(schedule) == max_iters
    return schedule    