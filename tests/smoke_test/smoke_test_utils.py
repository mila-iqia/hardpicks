import os


def get_directories(tmpdir):
    dir_dict = dict(
        data=os.path.join(tmpdir, "data"),
        output=os.path.join(tmpdir, "output"),
        mlflow_output=os.path.join(tmpdir, "mlflow_output"),
        tb_output=os.path.join(tmpdir, "tb_output"),
    )

    for dir in dir_dict.values():
        os.makedirs(dir)
    return dir_dict
