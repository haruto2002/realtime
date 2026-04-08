import datetime
import glob
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import wandb


class Saver(object):
    def __init__(
        self,
        cfg,
    ):
        self.cfg = cfg
        self.outputs_file_path = self.cfg.out_dir + "outputs.csv"

        self.results_list = []
        self.start_time = time.time()
        self.best_mae = 1e8
        if self.cfg.default.wandb:
            self.build_wandb()

    def build_wandb(self):
        wandb.init(
            # set the wandb project where this run will be logged
            project=self.cfg.wandb.project_name,
            name=f"P2P_{self.cfg.network.init_weight}_{self.cfg.dataset.name}_{self.cfg.wandb.run_name}",
        )

    def save_weights(self, model, epochs, val_mae):
        output_weight_dir_path = self.cfg.out_dir + "weights/"
        os.makedirs(output_weight_dir_path, exist_ok=True)

        # Remove past weights
        if not self.cfg.network.all_save:
            past_weights_path = sorted(
                glob.glob(output_weight_dir_path + "check_points_epochs_*")
            )
            for path in past_weights_path:
                os.remove(path)

        # Save latest weights
        save_weight_path = (
            output_weight_dir_path + f"check_points_epochs_{epochs:04}.pth"
        )
        torch.save(model.module.state_dict(), save_weight_path)

        # Save Best weight
        if self.best_mae > val_mae:
            self.best_mae = val_mae
            save_weight_path = output_weight_dir_path + "best_weights.pth"
            torch.save(model.module.state_dict(), save_weight_path)

    def save_results(
        self,
        epoch,
        train_gather_results,
        val_gather_results,
        lr,
    ):

        results = list(train_gather_results) + list(val_gather_results)
        self.results_list.append([epoch] + results + [lr])
        self.columns = [
            "epoch",
            "train_total_loss",
            "train_class_loss",
            "train_location_loss",
            "train_mae",
            "train_mse",
            "val_total_loss",
            "val_class_loss",
            "val_location_loss",
            "val_mae",
            "val_mse",
            "lr",
        ]

        self.results_df = pd.DataFrame(
            np.array(self.results_list),
            columns=self.columns,
        )
        self.results_df.to_csv(self.outputs_file_path, index=False)

        self.best_val_mae = self.results_df["val_mae"].values[-1]

        # save to wandb
        if self.cfg.default.wandb:
            dict_for_wandb = dict(
                [(key, self.results_df[key].values[-1]) for key in self.columns[1:]]
            )
            wandb.log(dict_for_wandb)

    def save_run_time(self):
        end_time = time.time()
        with open(self.cfg.out_dir + "run_time.txt", "w") as f:
            f.write(str(end_time - self.start_time))

        if self.cfg.default.wandb:
            wandb.finish()

    def show_results(self):

        index = len(self.results_df) - 1
        print(
            f"Epochs: [{int(self.results_df['epoch'][index])}/{self.cfg.default.epochs}]  "
            + f"Train Loss: {self.results_df['train_total_loss'][index]:.03f} "
            + f"(class: {self.results_df['train_class_loss'][index]:.02f}, loc: {self.results_df['train_location_loss'][index]:.02f}) "
            + f"Val Loss: {self.results_df['val_total_loss'][index]:.02f} "
            + f"(class: {self.results_df['val_class_loss'][index]:.02f}, loc: {self.results_df['val_location_loss'][index]:.02f})  "
            + f"Val MAE: {self.results_df['val_mae'][index]:.03f}  "
            + f"Val MSE: {self.results_df['val_mse'][index]:.03f}  "
            + f"Best MAE: {self.best_mae:03f}  "
            + str(datetime.datetime.now().time())
        )

    def plot_log(self):

        fig, axs = plt.subplots(figsize=(16, 9), nrows=2, ncols=3, sharex=True)

        axs = axs.ravel()
        axs[0].plot(
            self.results_df["train_total_loss"], label="Train", linewidth=4, alpha=0.8
        )
        axs[0].plot(
            self.results_df["val_total_loss"],
            label="Validation",
            linewidth=4,
            alpha=0.8,
        )
        axs[0].set_title("Total Loss", fontsize=20)
        # axs[0].set_ylim(
        #     min(
        #         self.results_df["train_total_loss"].min(),
        #         self.results_df["val_total_loss"].min(),
        #     ),
        #     np.percentile(self.results_df["val_total_loss"].values, 75),
        # )
        axs[0].grid()
        axs[0].legend()

        axs[1].plot(
            self.results_df["train_class_loss"], label="Train", linewidth=4, alpha=0.8
        )
        axs[1].plot(
            self.results_df["val_class_loss"],
            label="Validation",
            linewidth=4,
            alpha=0.8,
        )
        axs[1].set_title("Classification Loss", fontsize=20)
        # axs[1].set_ylim(
        #     self.results_df["train_class_loss"].min(),
        #     np.percentile(self.results_df["val_class_loss"].values, 75),
        # )
        axs[1].grid()
        axs[1].legend()

        axs[2].plot(
            self.results_df["train_location_loss"] * self.cfg.network.point_loss_coef,
            label="Train",
            linewidth=4,
            alpha=0.8,
        )
        axs[2].plot(
            self.results_df["val_location_loss"] * self.cfg.network.point_loss_coef,
            label="Validation",
            linewidth=4,
            alpha=0.8,
        )
        axs[2].set_title("Localization Loss", fontsize=20)
        # axs[2].set_ylim(
        #     self.results_df["train_location_loss"].min(),
        #     np.percentile(self.results_df["val_location_loss"].values, 75),
        # )
        axs[2].grid()
        axs[2].legend()

        axs[3].plot(self.results_df["train_mae"], label="Train", linewidth=4, alpha=0.8)
        axs[3].plot(
            self.results_df["val_mae"], label="Validation", linewidth=4, alpha=0.8
        )
        axs[3].set_title("MAE", fontsize=20)
        # axs[3].set_ylim(
        #     self.results_df["train_mae"].min(),
        #     np.percentile(self.results_df["val_mae"].values, 75),
        # )
        axs[3].grid()
        axs[3].legend()

        axs[4].plot(self.results_df["train_mse"], label="Train", linewidth=4, alpha=0.8)
        axs[4].plot(
            self.results_df["val_mse"], label="Validation", linewidth=4, alpha=0.8
        )
        axs[4].set_title("MSE", fontsize=20)
        # axs[4].set_ylim(
        #     self.results_df["train_mse"].min(),
        #     np.percentile(self.results_df["val_mse"].values, 75),
        # )
        axs[4].grid()
        axs[4].legend()

        axs[5].plot(self.results_df["lr"], linewidth=4, alpha=0.8)
        axs[5].set_title("Lerning Ratio", fontsize=20)
        # axs[4].set_ylim(
        #     self.results_df["train_mse"].min(),
        #     np.percentile(self.results_df["val_mse"].values, 75),
        # )
        axs[5].grid()

        fig.supxlabel("Epochs", fontsize=20)
        plt.tight_layout()
        plt.savefig(self.cfg.out_dir + "outputs.png")
        plt.clf()
        plt.close()
