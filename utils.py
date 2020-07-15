def soft_thresholding(a, b):
    if a > b:
        return a - b
    elif a < -b:
        return a + b
    else:
        return 0