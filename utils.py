def if_OOM(exception: str):
    return "CUDA out of memory" in exception

def simplify_exception(exception: str):
    if if_OOM(exception):
        return "Out of memory."
    else:
        return exception
    
def get_model_name(model: str):
    return model.split("/")[1]