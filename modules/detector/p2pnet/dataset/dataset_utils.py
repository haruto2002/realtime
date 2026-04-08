import os
import subprocess


def copy_datasets(cfg):
    if cfg.default.resource in ["ABCI", "Tsukuba"]:
        if not os.path.exists(f"{os.environ['SGE_LOCALDIR']}/dataset/{cfg.dataset.name}/"):
            os.makedirs(f"{os.environ['SGE_LOCALDIR']}/dataset/", exist_ok=True)
            if cfg.default.resource == "ABCI":
                if cfg.network.dnn_task == "counting":
                    copy_dataset_path = f"/groups/gcc50570/MLDatasets/CrowdCounting/{cfg.dataset.name}.tar.gz"
                elif cfg.network.dnn_task == "hanabi":
                    copy_dataset_path = f"/groups/gcc50570/MLDatasets/Hanabi/{cfg.dataset.name}.tar.gz"

            elif cfg.default.resource == "Tsukuba":
                if cfg.network.dnn_task == "counting":
                    copy_dataset_path = f"/homes/SHARE/MLDatasets/CrowdCounting/{cfg.dataset.name}.tar.gz"
                elif cfg.network.dnn_task == "hanabi":
                    copy_dataset_path = f"/homes/SHARE/MLDatasets/Hanabi/dataset/{cfg.dataset.name}.tar.gz"

            if not os.path.exists(copy_dataset_path):
                print(copy_dataset_path)
                raise ValueError("No such dataset exists!")
            subprocess.run(
                [
                    "cp",
                    copy_dataset_path,
                    f"{os.environ['SGE_LOCALDIR']}/dataset/",
                ]
            )
            subprocess.run(
                [f"tar -I  pigz -xf {cfg.dataset.name}.tar.gz"],
                cwd=f"{os.environ['SGE_LOCALDIR']}/dataset/",
                shell=True,
            )
            print("Copy dataaset to SGE_LOCALDIR !!")


def suggest_dataset_root_dir(cfg):
    if cfg.default.resource in ["ABCI", "Tsukuba"]:
        dataset_path = f"{os.environ['SGE_LOCALDIR']}/dataset/{cfg.dataset.name}/"
    elif cfg.default.resource == "local":
        dataset_path = f"./datasets/{cfg.dataset.name}/"
    else:
        raise ValueError("Resource type is selectd from 'ABCI', 'Tsukuba' or 'local'")

    if not os.path.exists(dataset_path):
        print(dataset_path)
        raise ValueError("No such dataset exists!")

    return dataset_path
