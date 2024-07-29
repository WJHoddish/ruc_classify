import re
import logging
import pandas as pd

from config import *


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


logger = logging.getLogger(__name__)
logger.info("preprocessing ...")


# get cell subsets
metadata_df = pd.read_csv("./data/LDX_all_meta.csv")
subsets = {
    i: metadata_df[
        metadata_df["Big_celltypes"].isin(
            [
                i,
            ]
        )
    ]
    .drop_duplicates(subset=["Sub_celltypes"])["Sub_celltypes"]
    .to_list()
    for i in metadata_df["Big_celltypes"].drop_duplicates().to_list()
}


def get_data(package):
    """_summary_

    Args:
        package (_type_): _description_
    """

    for patient, stage in [(x, y) for x in patients for y in stages]:
        # 训练只关注pre
        if stage != "pre":
            continue

        prefix = f"{patient}_{stage}"

        def rename(cols):
            return [re.sub(r"[+ ]", ".", col) for col in cols]

        df = pd.read_csv(
            f"{package['path']}/{package['prefix']}{prefix}{package['surfix']}",
            index_col=0,
        )

        df.columns = rename(df.columns)

        # df = df[subsets["Epithelial"]]

        def policy_all_dim():
            """Basic policy

            Returns:
                _type_: _description_
            """

            # 按列求和
            df_ = df.sum(axis=0)

            # 归一化
            df_ = df_ / df_.sum()

            return df_[subsets["Epithelial"]]

        def policy_two_dim():
            df_ = policy_all_dim()
            ret = [
                df_["immuneEPI"] + df_["Differential"],
                df_["CYCLING"] + df_["STRESS"] + df_["Intermedian"] + df_["EMT"],
            ]


            return ret

        def output_policy():
            data = policy_all_dim()
            # print(data)
            return data.to_numpy()

            data = policy_two_dim()
            # TODO: scatter plot

            return data

        yield prefix, output_policy()
