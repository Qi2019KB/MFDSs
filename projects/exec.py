# -*- coding: utf-8 -*-
from projects.FixMatch import exec as FixMatch
from projects.MFDSs import exec as MFDSs


def execute():
    dataParams = [["CIFAR10", 40], ["CIFAR10", 250], ["CIFAR10", 4000],
                  ["CIFAR100", 400], ["CIFAR100", 2500], ["CIFAR100", 10000]]
    for dataParam in dataParams:
        dataset, trainCount_labeled = dataParam

        # FixMatch
        FixMatch("FixMatch", {"dataset": dataset, "trainCount_labeled": trainCount_labeled, "useConsistency": False})

        # MFDS (FixMatch + Trusted Consistency loss)
        FixMatch("MFDS", {"dataset": dataset, "trainCount_labeled": trainCount_labeled})

        # MFDSs (without Feature Discrepant loss)
        MFDSs("MFDSs_noFDL", {"dataset": dataset, "trainCount_labeled": trainCount_labeled})

        # MFDSs
        MFDSs("MFDSs", {"dataset": dataset, "trainCount_labeled": trainCount_labeled, "useFDL": True})
    pass


if __name__ == "__main__":
    execute()
